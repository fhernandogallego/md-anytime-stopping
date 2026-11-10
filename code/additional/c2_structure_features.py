#!/usr/bin/env python3
"""Calculation 2 (part A): structure-only descriptors from the ATLAS starting structure.

Everything is computed from the static input PDB of each system, never from a trajectory
frame, so the resulting predictor is a genuine structure-only baseline. The model fitting
uses these features in c3_sweep.py (--structure-features).

Descriptors per system
  n_residues, sequence composition (hydrophobic, charged, polar, Gly, Pro, aromatic),
  CA radius of gyration and Rg / N^(1/3), total SASA (Shrake-Rupley), DSSP helix / strand /
  coil fractions, relative contact order and mean CA contacts per residue (8 A cut-off),
  end-to-end CA distance, and mean / std of B-factors when the file has non-zero values.

Usage
  python c2_structure_features.py --run /path/ATLAS942_run --pdb-glob "/path/atlas/{system}/*.pdb" \
      --out resultados/c2_structure_features.csv
  (use --pdb-glob with {system} replaced by each system_id; check the first lines of the output)
  python c2_structure_features.py --pdb-list systems_pdb.csv --out ...   (columns: system_id,pdb_path)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from common import read_csv, write_csv, load_folds

HYDROPHOBIC = set("AVILMFWC")
CHARGED = set("DEKR")
POLAR = set("STNQHY")
AROMATIC = set("FWYH")


def bfactors(path: Path) -> np.ndarray:
    vals = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                try:
                    vals.append(float(line[60:66]))
                except ValueError:
                    pass
    return np.asarray(vals)


def describe(path: Path) -> dict:
    import mdtraj as md

    traj = md.load(str(path))
    protein = traj.topology.select("protein")
    traj = traj.atom_slice(protein)
    seq = "".join(r.code or "X" for r in traj.topology.residues)
    n = len(seq)
    ca = traj.topology.select("name CA")
    xyz = traj.xyz[0, ca] * 10.0  # nm -> A
    rg = float(np.sqrt(np.mean(np.sum((xyz - xyz.mean(0)) ** 2, axis=1))))
    sasa = float(md.shrake_rupley(traj, mode="residue").sum() * 100.0)
    ss = md.compute_dssp(traj, simplified=True)[0]
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    i, j = np.triu_indices(len(ca), k=3)
    contact = d[i, j] < 8.0
    rco = float(np.mean(np.abs(i[contact] - j[contact])) / len(ca)) if contact.any() else 0.0
    b = bfactors(path)
    has_b = b.size and np.any(b > 0) and np.std(b) > 0
    return {
        "n_residues": n,
        "frac_hydrophobic": sum(c in HYDROPHOBIC for c in seq) / n,
        "frac_charged": sum(c in CHARGED for c in seq) / n,
        "frac_polar": sum(c in POLAR for c in seq) / n,
        "frac_gly": seq.count("G") / n,
        "frac_pro": seq.count("P") / n,
        "frac_aromatic": sum(c in AROMATIC for c in seq) / n,
        "rg_ca_A": rg,
        "rg_over_n13": rg / n ** (1 / 3),
        "sasa_total_A2": sasa,
        "sasa_per_residue_A2": sasa / n,
        "helix_fraction": float(np.mean(ss == "H")),
        "strand_fraction": float(np.mean(ss == "E")),
        "coil_fraction": float(np.mean(ss == "C")),
        "relative_contact_order": rco,
        "ca_contacts_per_residue": float(2 * contact.sum() / len(ca)),
        "end_to_end_A": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "bfactor_mean": float(b.mean()) if has_b else float("nan"),
        "bfactor_std": float(b.std()) if has_b else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="run folder with outer_folds.csv (defines the systems)")
    ap.add_argument("--pdb-glob", help='pattern with {system}, e.g. "/data/atlas/{system}/{system}.pdb"')
    ap.add_argument("--pdb-list", type=Path, help="CSV with system_id,pdb_path")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    if args.pdb_list:
        mapping = {r["system_id"]: Path(r["pdb_path"]) for r in read_csv(args.pdb_list)}
    else:
        if not (args.run and args.pdb_glob):
            ap.error("give --pdb-list, or --run and --pdb-glob")
        mapping = {}
        for system in sorted(load_folds(args.run)):
            hits = sorted(glob.glob(args.pdb_glob.format(system=system)))
            if len(hits) != 1:
                raise FileNotFoundError(f"{system}: {len(hits)} PDB files match; refine --pdb-glob ({hits[:3]})")
            mapping[system] = Path(hits[0])

    rows, failed = [], []
    for k, (system, path) in enumerate(sorted(mapping.items())):
        try:
            rows.append({"system_id": system, "pdb_path": str(path), **describe(path)})
        except Exception as exc:  # keep going, report at the end
            failed.append((system, repr(exc)))
        if k < 3 or (k + 1) % 100 == 0:
            print(f"[{k + 1}/{len(mapping)}] {system}", flush=True)
    print(f"systems={len(rows)} failed={len(failed)}")
    for system, err in failed:
        print(f"FAILED {system}: {err}")
    if not rows:
        raise SystemExit("No system could be described; see the errors above")
    write_csv(args.out, rows)
    b_ok = sum(np.isfinite(r["bfactor_mean"]) for r in rows)
    print(f"with_bfactors={b_ok}")
    print("C2_STRUCTURE_FEATURES=PASS" if not failed else "C2_STRUCTURE_FEATURES=PARTIAL")


if __name__ == "__main__":
    main()
