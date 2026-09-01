# OPNs-HybridBoost

## Paper Code & Reproducibility

**OPNs-HybridBoost: Algebraic Pairwise Interactions with Oblivious-Tree Refinement for Tabular Learning**

This subdirectory is the public research package for the OPNs-HybridBoost paper.

> The repository root remains the shared OPNs / OPNs-LR project. HybridBoost-specific reproducibility material is intentionally isolated here under `research/hybridboost/`.

[← Back to the repository homepage](../../README.md)

## Quick Start

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The root `requirements.txt` provides the shared OPNs / OPNs-LR environment and the core numerical dependencies used by HybridBoost.

For the complete Overall baseline comparison, install the optional external boosting libraries:

```bash
python -m pip install xgboost lightgbm catboost
```

`HistGradientBoosting` is provided by scikit-learn.

Before running expensive experiments, validate the encoded protocols and frozen configuration hashes:

```bash
python research/hybridboost/scripts/reproduce_overall.py --dry-run
python research/hybridboost/scripts/reproduce_warm_start.py --dry-run
python research/hybridboost/scripts/reproduce_active.py --dry-run
python research/hybridboost/scripts/reproduce_active.py --experiment structure --dry-run
```

Each dry-run writes protocol/environment metadata without fitting the models.

## Public Reproduction Entry Points

### 1. Overall benchmark

```bash
python research/hybridboost/scripts/reproduce_overall.py
```

Entry point: [`scripts/reproduce_overall.py`](./scripts/reproduce_overall.py)

Purpose: reproduce the final OPNs-HybridBoost comparison against the supported classic boosting baselines.

Public-runner defaults:

- classification datasets: `breast_cancer`, `car`, `iris`, `wine`;
- regression datasets: `airfoil`, `boston`, `concrete`, `energy_cooling`, `wine_quality`;
- models: `opns`, `xgboost`, `lightgbm`, `catboost`, `histgb`;
- 5 folds;
- 1 repeat;
- `random_state=161803`;
- Wine Quality uses training-fold-only preprocessing/imputation where required by the migrated protocol.

Frozen configurations:

- [`configs/classification_final.json`](./configs/classification_final.json)
- [`configs/regression_final.json`](./configs/regression_final.json)
- [`configs/classification_baselines_final.json`](./configs/classification_baselines_final.json)
- [`configs/regression_baselines_final.json`](./configs/regression_baselines_final.json)

### 2. Warm-start ablation

```bash
python research/hybridboost/scripts/reproduce_warm_start.py
```

Entry point: [`scripts/reproduce_warm_start.py`](./scripts/reproduce_warm_start.py)

Purpose: reproduce the warm-start mechanism study comparing:

- `full_all`: complete OPNs-HybridBoost with Phase-I warm start and all pair features available to the tree stage;
- `tree_only_all`: tree-only counterpart using the matched all-feature search route.

Public-runner defaults:

- classification datasets: `breast_cancer`, `wine`, `car`, `iris`;
- regression datasets: `airfoil`, `concrete`, `energy_cooling`, `wine_quality`;
- modes: `full_all`, `tree_only_all`;
- 5 folds;
- 1 repeat;
- `random_state=161803`;
- `checkpoint_step=10`;
- classification round budget capped at 60;
- staged predictions/probabilities are obtained from a single fitted final-budget model rather than retraining at every checkpoint.

Frozen configurations:

- [`configs/classification_final.json`](./configs/classification_final.json)
- [`configs/regression_warm_start_final.json`](./configs/regression_warm_start_final.json)

Primary curve artifacts include `raw_curve.csv`, `curve_summary.csv`, and paired warm-start comparison CSVs.

### 3. Active efficiency

The Active runner defaults to the efficiency experiment:

```bash
python research/hybridboost/scripts/reproduce_active.py
```

Equivalent explicit form:

```bash
python research/hybridboost/scripts/reproduce_active.py --experiment efficiency
```

Entry point: [`scripts/reproduce_active.py`](./scripts/reproduce_active.py)

Purpose: compare:

- `full_all`: Phase-I warm start with all pair features used by the tree search;
- `full_active`: the same HybridBoost route with tree search restricted to the Phase-I active feature set.

Formal public-runner defaults:

- regression only;
- datasets: `energy_cooling`, `wine_quality`;
- modes: `full_all`, `full_active`;
- 5 folds;
- 1 repeat;
- `random_state=161803`;
- `checkpoint_step=10`;
- frozen config: [`configs/regression_warm_start_final.json`](./configs/regression_warm_start_final.json).

Primary outputs:

- `raw_curve.csv`
- `curve_summary.csv`
- `active_pairwise_curve.csv`
- `active_pairwise_summary.csv`

The pairwise artifacts are constructed directly from the raw curve by pairing `full_all` and `full_active` on dataset, split, and tree checkpoint.

### 4. Phase-I active-set structure snapshot

```bash
python research/hybridboost/scripts/reproduce_active.py --experiment structure
```

This protocol is intentionally **not** a boosting learning curve. It records the Phase-I active-set structure with a zero tree budget.

Formal public-runner defaults:

- regression only;
- datasets: `airfoil`, `concrete`, `energy_cooling`, `wine_quality`;
- mode: `full_active`;
- 5 folds;
- 1 repeat;
- `random_state=161803`;
- `tree_budget_cap=0`;
- `phase1_enabled=True`;
- `tree_feature_mode=active`.

Primary outputs:

- `phase1_structure_snapshot.csv`
- `phase1_structure_summary.csv`

These files report the candidate pair count, active tree-search feature count, active-retention ratio, and Phase-I timing diagnostics.

## Reproduction Map

| Study | Script | Main comparison / object | Primary purpose |
|---|---|---|---|
| Overall | `reproduce_overall.py` | OPNs-HybridBoost vs. classic boosting baselines | Final predictive comparison |
| Warm-start | `reproduce_warm_start.py` | `full_all` vs. `tree_only_all` | Isolate the Phase-I warm-start contribution |
| Active efficiency | `reproduce_active.py --experiment efficiency` | `full_all` vs. `full_active` | Measure active-set search reduction, runtime, and predictive trade-off |
| Phase-I structure | `reproduce_active.py --experiment structure` | `full_active`, zero tree budget | Inspect active-set retention before tree refinement |

## Formal Protocol Conventions

Across the migrated public runners:

- final experiment randomness uses `random_state=161803`;
- formal evaluation uses 5 folds and one repeat unless a runner explicitly states otherwise;
- configuration SHA-256 values are encoded/recorded to detect accidental config drift;
- metadata includes `protocol.json` and `environment.json`;
- raw fold/checkpoint-level tables are treated as the primary reproducibility artifacts;
- summary standard deviations use the sample standard deviation (`ddof=1`);
- smoke/debug flags may reduce folds or tree budgets but do not redefine the formal protocol.

## Dataset Protocol Notes

The experiment loaders are implemented in [`../../opns_boost/data.py`](../../opns_boost/data.py).

Important migrated protocol details include:

- **Wine Quality:** fold-local preprocessing is performed using training-fold information only where required, preventing preprocessing leakage.
- **Energy Efficiency:** the repository includes `dataset/energy_efficiency_data.csv`, from which the supported `energy_heating` / `energy_cooling` targets are selected by the shared loader.
- **Car Evaluation:** the shared migrated data protocol preserves the corrected encoding used by the final repository experiments.

The related behavior is covered by repository tests.

## Frozen Configurations

```text
research/hybridboost/configs/
├── classification_final.json
├── regression_final.json
├── regression_warm_start_final.json
├── classification_baselines_final.json
└── regression_baselines_final.json
```

These files are the public final configurations used by the migrated reproduction routes. The reproduction scripts record their paths and hashes in generated metadata.

## Public Ablation Components

```text
research/hybridboost/ablations/
└── tree_only.py
```

[`ablations/tree_only.py`](./ablations/tree_only.py) provides the public tree-only regression ablation used by the warm-start reproduction path.

The frozen warm-start helpers are located under [`scripts/`](./scripts/) and are shared by the public warm-start/active entry points.

## Output and Metadata

The runners create experiment-specific output directories containing combinations of:

- `protocol.json` — encoded experiment protocol;
- `environment.json` — Python/platform/package and CLI metadata;
- frozen configuration snapshots;
- raw fold-level or fold/checkpoint-level CSV files;
- aggregated summary CSV files;
- paired comparison CSV files where the protocol calls for matched comparisons.

The raw tables should be treated as the primary evidence. Summary files are derived from those raw records.

## Smoke / Debug Runs

The public scripts expose CLI overrides for small checks. For example, the Active efficiency runner supports a one-fold, reduced-tree smoke run:

```bash
python research/hybridboost/scripts/reproduce_active.py \
  --experiment efficiency \
  --datasets energy_cooling \
  --max-folds 1 \
  --tree-budget-cap 10
```

These overrides are intended for installation/E2E verification only. Results from reduced smoke protocols should not be reported as the formal paper results.

## Tests

From the repository root:

```bash
python -m pytest -q
```

Relevant tests cover:

- frozen configuration contracts;
- shared OPNs compatibility;
- migrated dataset protocols;
- HybridBoost Phase-I and core smoke behavior;
- tree-only ablation semantics;
- frozen warm-start helper behavior;
- Overall/Warm-start/Active reproduction contracts and dry-runs.

## Directory Layout

```text
research/hybridboost/
├── README.md
├── ablations/
│   └── tree_only.py
├── configs/
│   ├── classification_baselines_final.json
│   ├── classification_final.json
│   ├── regression_baselines_final.json
│   ├── regression_final.json
│   └── regression_warm_start_final.json
└── scripts/
    ├── _classification_baselines.py
    ├── _regression_baselines.py
    ├── _warm_start_classification.py
    ├── _warm_start_regression.py
    ├── reproduce_active.py
    ├── reproduce_overall.py
    └── reproduce_warm_start.py
```

## Scope of the Public Research Package

This directory intentionally contains the **final public reproduction surface**, not every exploratory or one-off development script used during model development.

Exploratory representation-ladder, spectral-route, algebra-attribution, patching, profiling, and preflight scripts are not part of the formal public paper reproduction contract. The public surface is centered on the final Overall, Warm-start, and Active studies described above.

## Citation

If you use OPNs-HybridBoost in academic work, please cite:

**Yi Zheng, Hao Feng, Xiaoqin Pan, and Lei Zhou.**
*OPNs-HybridBoost: Algebraic Pairwise Interactions with Oblivious-Tree Refinement for Tabular Learning.*

For the underlying OPNs algebraic framework, also cite the foundational OPNs work listed below. Repository-level software citation metadata is available in [`../../CITATION.cff`](../../CITATION.cff).

## License

The OPNs-HybridBoost source code in this repository follows the repository-level [MIT License](../../LICENSE), unless otherwise noted.

Datasets, manuscript files, and third-party materials are not automatically covered by the MIT License and remain subject to their respective original licenses or copyright terms.

## Foundational OPNs References

The HybridBoost implementation builds on the OPNs framework introduced and developed in:

1. Lei Zhou. “Ordered pair of normalized real numbers.” *Information Sciences*, 538 (2020), 290–313.
   https://doi.org/10.1016/j.ins.2020.05.036

2. Lei Zhou. “Smith Normal Forms and Matrix Theory over Ordered Pair of Normalized Real Numbers.” 2026. Preprint.
   https://doi.org/10.20944/preprints202606.1206.v1
