# When Can a Molecular-Dynamics Simulation Stop?

Companion repository for the preprint

> Hernando-Gallego F, García-Santos D, Vasilev-Vasilev D, Gómez-Gómez A, Campo-Camacho A.
> *When Can a Molecular-Dynamics Simulation Stop? Toward Anytime-Valid Precision for Equilibrium Observables.*
> ChemRxiv (2026). DOI: to be added on publication.

The preprint formulates data-dependent stopping of MD simulations as an anytime-valid inference problem and reports exploratory first-stage results: forecasting the future-only mean RMSD of ATLAS replicas from a 20 ns prefix, with sequence-grouped cross-validation on cohorts of up to 757 evaluated systems.

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `references.bib`, `main.bbl`), compiled `main.pdf` and the figure |
| `results/cohort_metrics.csv` | Out-of-fold MAE, RMSE and R² per cohort and model (Table 2) |
| `results/paired_contrasts.csv` | Paired MAE differences with 95% group-bootstrap intervals (Table 3, Figure 1) |
| `results/atlas942/` | System-level metrics, paired-bootstrap audit and run summary for the largest cohort, as produced by the pipeline |
| `figures/make_fig1.py` | Redraws Figure 1 from `results/paired_contrasts.csv` |
| `code/` | Snapshot of the training and evaluation code used for the ATLAS942 run (future-only target builder, ridge on 48 prefix summaries, residual GRU, baselines, fold worker and full refit) |
| `results/atlas942_additional/` | Additional ATLAS942 analyses for the journal version (see below) |
| `code/additional/` | Scripts that produced them: structure descriptors, multi-observable targets, prefix-length sweep and cross-conformal intervals |

## Additional ATLAS942 analyses (November 2026)

Run on the same outer folds, sequence groups and replica masks as the ATLAS942 cohort, without evaluating validation or test predictions. Ridge with the original penalty grid reproduces the reported RMSD MAE (0.610037 Å) in `regression_check/`.

| Path | Content |
|---|---|
| `structure_features.csv` | 17 descriptors of the starting ATLAS structure of each chain (no trajectory frames) |
| `forecast_<observable>/` | Out-of-fold metrics, paired contrasts, fold-wise selections and system-level predictions for persistence, prefix mean, ridge (extended grid), structure-only ridge, prefix + structure ridge and, for RMSD and core RMSD, the residual GRU, with 5, 10 and 20 ns prefixes |
| `conformal_<observable>/` | Cross-conformal coverage and width (absolute, group-max and, for RMSD, scaled scores) and coverage by quartile |
| `c3_targets.summary.json` | Valid replicas and systems per observable for the future-only targets |
| `c3a_zero_variance_fix.json` | Constant future windows (11 helix, 350 strand) where PyMBAR's inefficiency is undefined; g = 1 is used as a computational convention |

Observables: backbone RMSD, replica-specific core RMSD (30th RMSF quantile, core defined on the prefix), radius of gyration, total SASA, and helix and strand fractions.

Redraw the figure with:

```bash
python3 figures/make_fig1.py
```

## Full evidence bundles

Fold manifests, per-system predictions, checksums and run metadata for each cohort are archived in
[`md-hpc-benchmarks-public`](https://github.com/fhernandogallego/md-hpc-benchmarks-public) at pinned commits:

| Cohort | Commit |
|---|---|
| ATLAS222 | [`260b8e0`](https://github.com/fhernandogallego/md-hpc-benchmarks-public/tree/260b8e030461676c14b7e2f5029f4ccbc5ca1552/releases/paper2_atlas222_20261016_v1) |
| ATLAS362 | [`0a97696`](https://github.com/fhernandogallego/md-hpc-benchmarks-public/tree/0a97696df7cee24e8e18e8b4cbfe57865bf2de84/reports/paper2_atlas362_cv_20261020) |
| ATLAS742 | [`554f4b6`](https://github.com/fhernandogallego/md-hpc-benchmarks-public/tree/554f4b68d57b0a563b28fad9e451f17e18f5e11d/reports/paper2_atlas742_cv_20261029) |
| ATLAS942 | [`aeac568`](https://github.com/fhernandogallego/md-hpc-benchmarks-public/tree/aeac56891e06bb002d50ebdd3a1068c84fd2c600/reports/paper2_atlas942_cv_20261103) |

## Scope and limitations

- The code snapshot documents what was run; it keeps the original module paths and is not a self-contained rerun package. Model weights are not included.
- No held-out validation or test predictions have been evaluated. Bootstrap intervals are conditional on fixed out-of-fold predictions.
- No stopping policy, anytime-valid coverage or computational saving is claimed.

## Data

ATLAS trajectories are available from their authors under CC BY-NC 4.0 ([Vander Meersche et al., 2024](https://doi.org/10.1093/nar/gkad1084)) and are not redistributed here. Metrics and predictions derived from ATLAS are shared for non-commercial research use under the same terms.

## Licence

Code (`code/`, `figures/`): MIT. Manuscript and documentation: CC BY 4.0. ATLAS-derived results: CC BY-NC 4.0.

## Contact

Francisco Hernando-Gallego, Universidad de Valladolid — fhernando@uva.es
