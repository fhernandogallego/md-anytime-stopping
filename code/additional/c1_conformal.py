#!/usr/bin/env python3
"""Calculation 1: group-aware conformal prediction intervals on existing out-of-fold predictions.

No model is retrained. For every outer fold k, calibration scores come from the out-of-fold
predictions of the other four folds (a cross-conformal / CV+ style construction), and intervals
are evaluated on fold k. Sequence groups never cross folds, so calibration and evaluation
systems belong to different groups.

Variants
  absolute   score = |true - predicted| per system
  group_max  score = max |true - predicted| within each sequence group (conservative; one
             exchangeable score per group)
  scaled     score = |true - predicted| / sigma_s, with sigma_s the prefix RMSD variability of
             the system (std over 15-20 ns, mean over replicas). Needs dataset.npz.

Usage
  python c1_conformal.py --run /path/to/ATLAS942_run --out resultados/c1_conformal
  (the run folder must contain system_predictions.csv and outer_folds.csv; dataset.npz and
   train_future_rmsd_rows.csv are used for the scaled variant when present)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import read_csv, write_csv, write_json, sha, load_prefix

LEVELS = (0.10, 0.05)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def group_coverage_ci(covered: np.ndarray, groups: np.ndarray, draws=2000, seed=20261108):
    names = sorted(set(groups))
    index = {g: np.flatnonzero(groups == g) for g in names}
    rng = np.random.default_rng(seed)
    vals = [covered[np.concatenate([index[names[i]] for i in rng.integers(0, len(names), len(names))])].mean()
            for _ in range(draws)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def prefix_sigma(run: Path, systems: list[str]) -> dict[str, float] | None:
    if not (run / "dataset.npz").is_file() or not (run / "train_future_rmsd_rows.csv").is_file():
        return None
    valid = {r["sample_id"] for r in read_csv(run / "train_future_rmsd_rows.csv")
             if r["future_valid"].strip().lower() in ("true", "1")}
    sample_ids, system_ids, x = load_prefix(run)
    per = {}
    for i, (sid, sysid) in enumerate(zip(sample_ids, system_ids)):
        if sid in valid:
            per.setdefault(sysid, []).append(float(np.std(x[i, 150:, 0])))
    sig = {s: float(np.mean(v)) for s, v in per.items()}
    floor = 0.1 * float(np.median(list(sig.values())))
    return {s: max(sig.get(s, np.nan), floor) for s in systems}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="defaults to RUN/system_predictions.csv")
    args = ap.parse_args()
    pred_path = args.predictions or args.run / "system_predictions.csv"
    rows = read_csv(pred_path)
    args.out.mkdir(parents=True, exist_ok=False)

    systems_all = sorted({r["system_id"] for r in rows})
    sigma_rmsd = prefix_sigma(args.run, systems_all)
    summary, strata = [], []
    # predictions from c3_sweep.py carry target and prefix_ns columns: one analysis per combination
    keys = sorted({(r.get("target", "rmsd"), r.get("prefix_ns", "20")) for r in rows})
    combos = [(t, p, m) for t, p in keys for m in sorted({r["model"] for r in rows
              if r.get("target", "rmsd") == t and r.get("prefix_ns", "20") == p})]

    for target, prefix, model in combos:
        sigma = sigma_rmsd if target == "rmsd" and float(prefix) == 20.0 else None
        mr = sorted((r for r in rows if r["model"] == model and r.get("target", "rmsd") == target
                     and r.get("prefix_ns", "20") == prefix), key=lambda r: r["system_id"])
        sys_ids = np.asarray([r["system_id"] for r in mr])
        groups = np.asarray([r["group_id"] for r in mr])
        folds = np.asarray([int(r["fold"]) for r in mr])
        true = np.asarray([float(r["true"]) for r in mr])
        pred = np.asarray([float(r["predicted"]) for r in mr])
        resid = np.abs(true - pred)
        variants = {"absolute": np.ones(len(mr)), "group_max": np.ones(len(mr))}
        if sigma is not None:
            variants["scaled"] = np.asarray([sigma[s] for s in sys_ids])

        for variant, scale in variants.items():
            for alpha in LEVELS:
                lo = np.empty(len(mr)); hi = np.empty(len(mr))
                for k in sorted(set(folds)):
                    cal, ev = folds != k, folds == k
                    if variant == "group_max":
                        g_scores = {}
                        for g, r in zip(groups[cal], resid[cal]):
                            g_scores[g] = max(g_scores.get(g, 0.0), r)
                        q = conformal_quantile(np.asarray(list(g_scores.values())), alpha)
                    else:
                        q = conformal_quantile(resid[cal] / scale[cal], alpha)
                    lo[ev] = pred[ev] - q * scale[ev]
                    hi[ev] = pred[ev] + q * scale[ev]
                covered = (true >= lo) & (true <= hi)
                width = hi - lo
                w_group = np.asarray([1.0 / np.sum(groups == g) for g in groups])
                ci = group_coverage_ci(covered.astype(float), groups)
                summary.append({
                    "target": target, "prefix_ns": prefix, "model": model, "variant": variant, "nominal": 1 - alpha,
                    "n_systems": len(mr), "n_groups": len(set(groups)),
                    "coverage_systems": float(covered.mean()),
                    "coverage_ci95_low": ci[0], "coverage_ci95_high": ci[1],
                    "coverage_group_weighted": float(np.sum(covered * w_group) / np.sum(w_group)),
                    "mean_width_A": float(width.mean()), "median_width_A": float(np.median(width)),
                    "coverage_min_fold": float(min(covered[folds == k].mean() for k in set(folds))),
                    "coverage_max_fold": float(max(covered[folds == k].mean() for k in set(folds))),
                })
                # conditional coverage by quartile of the true target and of sigma
                keys = {"true_quartile": true}
                if sigma is not None:
                    keys["prefix_sigma_quartile"] = np.asarray([sigma[s] for s in sys_ids])
                for name, values in keys.items():
                    edges = np.quantile(values, [0.25, 0.5, 0.75])
                    q_idx = np.searchsorted(edges, values, side="right")
                    for q in range(4):
                        m = q_idx == q
                        strata.append({"target": target, "prefix_ns": prefix, "model": model, "variant": variant, "nominal": 1 - alpha,
                                       "stratum": name, "quartile": q + 1, "n_systems": int(m.sum()),
                                       "coverage": float(covered[m].mean()),
                                       "mean_width_A": float(width[m].mean())})

    write_csv(args.out / "conformal_summary.csv", summary)
    write_csv(args.out / "conformal_by_stratum.csv", strata)
    write_json(args.out / "summary.json", {
        "calculation": "c1_conformal", "predictions": str(pred_path), "predictions_sha256": sha(pred_path),
        "scaled_variant": sigma_rmsd is not None, "levels": [1 - a for a in LEVELS],
        "method": "cross-conformal on out-of-fold predictions; calibration from the other outer folds",
    })
    for r in summary:
        print(f"{r['target']:13s} {r['prefix_ns']:>4} {r['model']:28s} {r['variant']:9s} nominal={r['nominal']:.2f} "
              f"coverage={r['coverage_systems']:.3f} [{r['coverage_ci95_low']:.3f},{r['coverage_ci95_high']:.3f}] "
              f"width={r['mean_width_A']:.3f}")
    print("C1_CONFORMAL=PASS")


if __name__ == "__main__":
    main()
