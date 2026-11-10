#!/usr/bin/env python3
"""Calculations 2 (part B) and 3: models by observable and prefix length, with a structure-only baseline.

For every target observable and prefix length (5, 10, 20 ns), on the outer folds of an existing
run (same sequence groups, same folds):
  persistence        last value of the same observable in the prefix
  prefix_mean        mean of the same observable over the prefix
  ridge_summary      ridge on 6 summaries x 8 channels, extended alpha grid (up to 1e4)
  structure_only     ridge on static descriptors of the starting PDB (c2), no trajectory input
  prefix_structure   ridge on prefix summaries + static descriptors
  gru                residual GRU (only for --gru-targets; channel of the target first)
Sanity check: for rmsd at 20 ns, ridge with the original alpha grid must reproduce the
published ATLAS942 ridge MAE (0.610037 A) when run on the ATLAS942 folder.

Usage
  python c3_sweep.py --run /path/ATLAS942_run --targets resultados/c3_targets.csv \
      --structure-features resultados/c2_structure_features.csv \
      --only rmsd --prefix-ns 5 10 20 --gru-targets rmsd core_rmsd_q30 --out resultados/c3_rmsd
  (one call per target is convenient on Slurm; see lanzar_c3.sbatch)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import (ALPHAS_EXTENDED, ALPHAS_ORIGINAL, SEED, group_bootstrap_delta, load_folds, load_prefix,
                    metrics, read_csv, ridge_cv_predict, run_gru, sha, summarize, write_csv, write_json)

CHANNEL = {"rmsd": 0, "rg": 1, "core_rmsd_q30": 2, "core_rmsd_q40": 3, "core_rmsd_q50": 4,
           "sasa": 5, "helix": 6, "strand": 7}
CONTRASTS = [("ridge_summary", "persistence"), ("structure_only", "persistence"),
             ("ridge_summary", "structure_only"), ("prefix_structure", "ridge_summary"),
             ("prefix_structure", "structure_only"), ("gru", "ridge_summary"), ("gru", "persistence")]


def load_structure(path: Path | None):
    if path is None:
        return None, []
    rows = read_csv(path)
    names = [c for c in rows[0] if c not in ("system_id", "pdb_path")]
    table = {r["system_id"]: np.asarray([float(r[c]) if r[c] not in ("", "nan") else np.nan for c in names]) for r in rows}
    mat = np.vstack(list(table.values()))
    keep = [i for i in range(len(names)) if np.isfinite(mat[:, i]).all()]
    return {s: v[keep] for s, v in table.items()}, [names[i] for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--structure-features", type=Path)
    ap.add_argument("--only", nargs="*", help="targets to run (default: all in --targets)")
    ap.add_argument("--prefix-ns", nargs="*", type=float, default=[5.0, 10.0, 20.0])
    ap.add_argument("--gru-targets", nargs="*", default=["rmsd", "core_rmsd_q30"])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--rmsd-from-run", action="store_true",
                    help="use RUN/train_future_rmsd_rows.csv for the rmsd target (exact published labels)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    folds = load_folds(args.run)
    sample_ids, system_ids, x_all = load_prefix(args.run)
    lookup = {s: i for i, s in enumerate(sample_ids)}
    struct, struct_names = load_structure(args.structure_features)
    target_rows = [r for r in read_csv(args.targets) if r["future_valid"].strip().lower() in ("true", "1")]
    if args.rmsd_from_run:
        target_rows = [r for r in target_rows if r["target"] != "rmsd"]
        for r in read_csv(args.run / "train_future_rmsd_rows.csv"):
            if r["future_valid"].strip().lower() in ("true", "1"):
                target_rows.append({"sample_id": r["sample_id"], "system_id": r["system_id"], "target": "rmsd",
                                    "future_valid": "True", "future_target": r["future_target_A"]})
    targets = args.only or sorted({r["target"] for r in target_rows})
    pred_rows, metric_rows, contrast_rows, selection = [], [], [], []

    for target in targets:
        if target not in CHANNEL:
            raise ValueError(f"No prefix channel for target {target}")
        rows = sorted((r for r in target_rows if r["target"] == target and r["system_id"] in folds
                       and r["sample_id"] in lookup), key=lambda r: r["sample_id"])
        imputed = []
        if struct is not None:  # systems without a PDB keep their rows; features = median (reported)
            median = np.median(np.vstack(list(struct.values())), axis=0)
            imputed = sorted({r["system_id"] for r in rows} - set(struct))
            for s_missing in imputed:
                struct[s_missing] = median
            if imputed:
                print(f"  WARNING {len(imputed)} systems without structure features (median imputed)", flush=True)
        idx = np.asarray([lookup[r["sample_id"]] for r in rows])
        y = np.asarray([float(r["future_target"]) for r in rows])
        systems = np.asarray([r["system_id"] for r in rows])
        groups = np.asarray([folds[s]["group_id"] for s in systems])
        fold = np.asarray([folds[s]["outer_fold"] for s in systems])
        ch = CHANNEL[target]
        order = [ch] + [c for c in range(8) if c != ch]
        x_full = x_all[idx][:, :, order]
        s_feat = np.vstack([struct[s] for s in systems]) if struct is not None else None
        print(f"{target}: {len(rows)} replicas, {len(set(systems))} systems, {len(set(groups))} groups", flush=True)

        for prefix in args.prefix_ns:
            npts = int(round(prefix / 0.1)) + 1
            x = x_full[:, :npts, :]
            summ = summarize(x)
            preds = {m: np.full(len(y), np.nan) for m in
                     ["persistence", "prefix_mean", "ridge_summary", "ridge_summary_original_grid",
                      "structure_only", "prefix_structure", "gru"]}
            t_start = time.time()
            for k in range(5):
                fit_idx, ev = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                assert set(groups[fit_idx]).isdisjoint(groups[ev])
                seed = SEED + 100 * k
                preds["persistence"][ev] = x[ev, -1, 0]
                preds["prefix_mean"][ev] = x[ev, :, 0].mean(1)
                p, a = ridge_cv_predict(summ, y, systems, groups, fit_idx, ev, ALPHAS_EXTENDED, seed)
                preds["ridge_summary"][ev] = p
                sel = {"target": target, "prefix_ns": prefix, "fold": k, "ridge_alpha": a}
                if prefix == 20.0:
                    p, a0 = ridge_cv_predict(summ, y, systems, groups, fit_idx, ev, ALPHAS_ORIGINAL, seed)
                    preds["ridge_summary_original_grid"][ev] = p
                    sel["ridge_alpha_original_grid"] = a0
                if s_feat is not None:
                    p, a = ridge_cv_predict(s_feat, y, systems, groups, fit_idx, ev, ALPHAS_EXTENDED, seed)
                    preds["structure_only"][ev] = p; sel["structure_alpha"] = a
                    both = np.hstack([summ, s_feat])
                    p, a = ridge_cv_predict(both, y, systems, groups, fit_idx, ev, ALPHAS_EXTENDED, seed)
                    preds["prefix_structure"][ev] = p; sel["prefix_structure_alpha"] = a
                if target in args.gru_targets:
                    p, e = run_gru(x, y, systems, groups, fit_idx, ev, SEED + k)
                    preds["gru"][ev] = p; sel["gru_epoch"] = e
                selection.append(sel)
            print(f"  prefix {prefix:g} ns done in {time.time() - t_start:.0f} s", flush=True)

            # system level
            sys_names = sorted(set(systems))
            sys_group = np.asarray([folds[s]["group_id"] for s in sys_names])
            sys_true = np.asarray([y[systems == s].mean() for s in sys_names])
            sys_err = {}
            for model, p in preds.items():
                if not np.isfinite(p).all():
                    continue
                sp = np.asarray([p[systems == s].mean() for s in sys_names])
                sys_err[model] = sp - sys_true
                metric_rows.append({"target": target, "prefix_ns": prefix, "model": model,
                                    "n_systems": len(sys_names), "n_groups": len(set(sys_group)),
                                    **metrics(sys_true, sp)})
                for s, t, q in zip(sys_names, sys_true, sp):
                    pred_rows.append({"target": target, "prefix_ns": prefix, "model": model, "system_id": s,
                                      "group_id": folds[s]["group_id"], "fold": folds[s]["outer_fold"],
                                      "true": float(t), "predicted": float(q)})
            best_naive = min(("persistence", "prefix_mean"), key=lambda m: np.abs(sys_err[m]).mean())
            pairs = CONTRASTS + ([("ridge_summary", best_naive)] if best_naive != "persistence" else [])
            for a, b in pairs:
                if a in sys_err and b in sys_err and a != b:
                    d, lo, hi = group_bootstrap_delta(sys_err[a], sys_err[b], sys_group, draws=args.draws)
                    contrast_rows.append({"target": target, "prefix_ns": prefix, "contrast": f"{a} - {b}",
                                          "delta_mae": d, "ci95_low": lo, "ci95_high": hi,
                                          "relative_change": d / float(np.abs(sys_err[b]).mean()),
                                          "excludes_zero": bool(lo > 0 or hi < 0)})

    write_csv(args.out / "system_predictions.csv", pred_rows)
    write_csv(args.out / "metrics.csv", metric_rows)
    write_csv(args.out / "contrasts.csv", contrast_rows)
    write_csv(args.out / "selection.csv", selection)
    write_json(args.out / "summary.json", {
        "calculation": "c3_sweep", "run": str(args.run), "targets_sha256": sha(args.targets),
        "structure_features": str(args.structure_features) if args.structure_features else None,
        "structure_feature_names": struct_names, "prefix_ns": args.prefix_ns, "targets": targets,
        "alphas_extended": ALPHAS_EXTENDED, "structure_imputed_last_target": imputed, "bootstrap_draws": args.draws,
        "validation_test_evaluated": False})
    for r in metric_rows:
        print(f"{r['target']:14s} {r['prefix_ns']:4g} ns {r['model']:28s} MAE={r['mae']:.6f} R2={r['r2']:.4f}")
    for r in contrast_rows:
        flag = "*" if r["excludes_zero"] else " "
        print(f"{r['target']:14s} {r['prefix_ns']:4g} ns {r['contrast']:36s} {r['delta_mae']:+.4f} "
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] {flag}")
    print("C3_SWEEP=PASS")


if __name__ == "__main__":
    main()
