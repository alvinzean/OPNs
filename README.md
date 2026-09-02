# OPNs: OPNs-LR Research Code and Core Algebra Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This repository originated as the implementation and reproduction codebase for OPNs-based linear regression (OPNs-LR). During the development of OPNs-LR, the project also established reusable software infrastructure for **Ordered Pair of Normalized Real Numbers (OPNs)**, including scalar algebra, mathematical functions, array and matrix operations, feature construction, preprocessing, and regression components. These shared OPNs components have since been reused and extended by other OPNs-based learning methods, including OPNs-HybridBoost.

> [!IMPORTANT]
> ## OPNs-HybridBoost — Paper Code & Reproducibility
>
> **If you arrived here from the OPNs-HybridBoost manuscript, go directly to:**
>
> ### ➜ [OPNs-HybridBoost project and reproduction guide](./research/hybridboost/)
>
> **Paper:** *OPNs-HybridBoost: Algebraic Pairwise Interactions with Oblivious-Tree Refinement for Tabular Learning*
>
> Public reproduction entry points:
>
> - [Overall benchmark](./research/hybridboost/scripts/reproduce_overall.py)
> - [Warm-start ablation](./research/hybridboost/scripts/reproduce_warm_start.py)
> - [Active efficiency / Phase-I structure](./research/hybridboost/scripts/reproduce_active.py)
>
> The full protocol, datasets, frozen configurations, and output descriptions are documented in [`research/hybridboost/README.md`](./research/hybridboost/README.md).

## Repository Scope

This repository was created during the development of OPNs-LR and has subsequently evolved into a shared research codebase for OPNs-based learning methods.

The root project therefore has three closely related roles:

1. **OPNs-LR implementation and reproduction** - the original primary purpose of the repository. The root `test.py`, `opns_module/linear_model`, `utils/configs`, and bundled regression datasets support this research workflow.
2. **Shared OPNs software infrastructure** - the reusable algebraic and numerical components developed alongside OPNs-LR, primarily under `opns_pack` and `opns_module`.
3. **Host repository for later OPNs research projects** - newer methods can reuse the shared OPNs implementation while keeping their paper-specific experimental material in dedicated subdirectories. OPNs-HybridBoost follows this structure under [`research/hybridboost/`](./research/hybridboost/).

The root-level workflow remains centered on OPNs-LR; paper-specific HybridBoost code and reproducibility material are intentionally isolated under `research/hybridboost/`.

## OPNs-LR — Primary Root Project

The root-level experiment workflow supports the OPNs linear-regression line of research associated with:

**Yi Zheng, Yonglin Huang, Xiaoqin Pan, Hui Zhang, Lei Zhou.**
*Multiple Linear Regression Based on the Framework of Ordered Pair of Normalized Real Numbers.*

A manuscript copy is available at [`OPNs-LR.pdf`](./OPNs-LR.pdf).

### Main OPNs-LR components

```text
test.py
opns_module/
├── linear_model/
└── preprocessing/
opns_pack/
utils/
└── configs/
    ├── config.json
    ├── default_params.json
    └── test_data_params.json
dataset/
requirements.txt
```

`test.py` builds OPNs feature representations, applies OPNs-aware preprocessing, runs the OPNs Lasso / linear-regression workflow, and reports repeated-run regression metrics.

### Install the root OPNs-LR environment

```bash
git clone https://github.com/alvinzean/OPNs.git
cd OPNs
python -m pip install -r requirements.txt
```

`requirements.txt` is the root OPNs / OPNs-LR dependency set. OPNs-HybridBoost has additional optional baseline dependencies documented separately in [`research/hybridboost/README.md`](./research/hybridboost/README.md).

### Run a bundled regression dataset

For example:

```bash
python test.py --dataset bike
```

Other bundled root datasets include `abalone`, `concrete`, `diabetes`, `energy_cooling`, `energy_heating`, `folds`, `wine`, and `yacht`.

Dataset-specific experiment parameters are stored in:

[`utils/configs/test_data_params.json`](./utils/configs/test_data_params.json)

When no dataset-specific entry is available, the root experiment code falls back to:

[`utils/configs/default_params.json`](./utils/configs/default_params.json)

To enable the root memory-analysis path:

```bash
python test.py --dataset bike --memory
```

### Using a custom regression dataset

The legacy/general root experiment entry point can also be extended to a custom dataset:

1. place the dataset in `dataset/`;
2. add an appropriate loader in `utils/functions.py` when the schema needs custom handling;
3. add dataset-specific parameters to `utils/configs/test_data_params.json`, or use the default configuration;
4. add the dataset name to the `--dataset` choices in `test.py`;
5. run `python test.py --dataset <dataset_name>`.

## Foundational OPNs Theory

The mathematical framework used by OPNs-LR, OPNs-HybridBoost, and the other OPNs-based algorithms in this repository originates from the following work by **Lei Zhou**.

1. **Lei Zhou.** “Ordered pair of normalized real numbers.” *Information Sciences*, 538 (2020), 290–313.
   DOI: [10.1016/j.ins.2020.05.036](https://doi.org/10.1016/j.ins.2020.05.036)

   This paper introduces the OPNs framework and develops its basic algebraic operations, ordering structure, elementary functions, and related theoretical properties.

2. **Lei Zhou.** “Smith Normal Forms and Matrix Theory over Ordered Pair of Normalized Real Numbers.” 2026. **Preprint**.
   DOI: [10.20944/preprints202606.1206.v1](https://doi.org/10.20944/preprints202606.1206.v1)

   This work further develops OPNs algebra and matrix theory, including the spectral two-channel representation, Smith normal forms, and related matrix results.

## Core OPNs Package

The reusable OPNs implementation developed as part of the OPNs-LR research codebase is primarily located in [`opns_pack/`](./opns_pack/). These components now serve as shared infrastructure for OPNs-LR and later OPNs-based algorithms.

### Scalar operations

```python
from opns_pack.opns import OPNs

a = OPNs(3, 4)
b = OPNs(-3, -4)

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
```

### Mathematical functions

```python
from opns_pack.opns import OPNs
from opns_pack import opns_math

a = OPNs(3, 4)

print(opns_math.log(a))
print(opns_math.sin(a))
print(opns_math.exp(a))
```

### Array and matrix operations

For NumPy-like OPNs operations, see:

- [`opns_pack/opns_np.py`](./opns_pack/opns_np.py)
- [`opns_pack/opns_matrix.py`](./opns_pack/opns_matrix.py)

These shared components are used by the regression code and can also serve other OPNs-based learning algorithms.

## OPNs-HybridBoost

OPNs-HybridBoost is maintained as a dedicated research package rather than being mixed into the root OPNs-LR workflow.

### ➜ [Open the OPNs-HybridBoost reproduction guide](./research/hybridboost/README.md)

The public package contains:

```text
research/hybridboost/
├── README.md
├── ablations/
├── configs/
└── scripts/
    ├── reproduce_overall.py
    ├── reproduce_warm_start.py
    └── reproduce_active.py
```

The three entry points cover the final Overall benchmark, the Warm-start mechanism study, and the Active efficiency / Phase-I structure study.

## Repository Structure

```text
OPNs/
├── opns_pack/                     # Shared OPNs scalar/math/array/matrix infrastructure
├── opns_module/                   # OPNs regression and preprocessing modules
├── test.py                        # Root OPNs-LR experiment entry point
├── utils/                         # Root OPNs-LR data utilities and experiment configs
├── dataset/                       # Bundled datasets
├── opns_boost/                    # HybridBoost implementation and shared experiment utilities
├── research/
│   └── hybridboost/               # OPNs-HybridBoost public research/reproduction package
├── tests/                         # Compatibility, protocol, and HybridBoost tests
├── OPNs-LR.pdf                    # OPNs-LR manuscript copy
├── requirements.txt               # Root OPNs / OPNs-LR dependencies
└── README.md
```

## OPNs-HybridBoost Dry-Run Checks

From the repository root, the formal HybridBoost protocols/configuration hashes can be checked without fitting models:

```bash
python research/hybridboost/scripts/reproduce_overall.py --dry-run
python research/hybridboost/scripts/reproduce_warm_start.py --dry-run
python research/hybridboost/scripts/reproduce_active.py --dry-run
python research/hybridboost/scripts/reproduce_active.py --experiment structure --dry-run
```

Full commands and protocol details are documented in the dedicated HybridBoost guide.

## Tests

Run the repository test suite with:

```bash
python -m pytest -q
```

The current test surface includes shared OPNs compatibility, migrated dataset protocols, HybridBoost core behavior, frozen configuration contracts, ablation semantics, and public reproduction contracts.

## Representative OPNs-Based Research

### OPNs-kNN

**Yi Zheng, Xuanbin Ding, Xiang Zhao, Xiaoqin Pan, Lei Zhou.**
“K-Nearest Neighbor Algorithm Based on the Framework of Ordered Pair of Normalized Real Numbers.”
*IEEE Transactions on Artificial Intelligence*, 2025.
DOI: [10.1109/TAI.2025.3566925](https://doi.org/10.1109/TAI.2025.3566925)

### OPNs K-means

**Ying Tang, Jia Guo, Yi Zheng, Hao Feng, Xiaoqin Pan, Lei Zhou.**
“K-means clustering with generalized metrics using Ordered Pair of Normalized real numbers.”
*Pattern Recognition*, 2026.
DOI: [10.1016/j.patcog.2026.114236](https://doi.org/10.1016/j.patcog.2026.114236)

### Other OPNs applications

- **Meijun Chen, Yi Zheng, Xiaoqin Pan, Lei Zhou.** “Generalized-Metric-Based Pattern Recognition Using Ordered Pair of Normalized Real Numbers.” *Applied Artificial Intelligence*, 2025. DOI: [10.1080/08839514.2025.2590815](https://doi.org/10.1080/08839514.2025.2590815)
- **Yonglin Huang, Yi Zheng, Xiaoqin Pan, Lei Zhou.** “Stepwise regression algorithm based on the ordered pair of normalized real numbers framework.” *The Journal of Supercomputing*, 2025. DOI: [10.1007/s11227-025-07369-6](https://doi.org/10.1007/s11227-025-07369-6)

## License

Unless otherwise noted, source code authored for this repository is licensed under the [MIT License](./LICENSE).

The MIT License applies to the repository's own software source code and associated software documentation. Dataset files, manuscript PDFs (including `OPNs-LR.pdf`), and third-party materials are **not automatically covered by the MIT License** and remain subject to their respective original licenses, terms of use, or copyright conditions.

Copyright is attributed collectively to **OPNs contributors**; individual contributors retain the rights associated with their respective contributions.

## Citation

This is a multi-project OPNs research repository. For academic use, please cite the publication corresponding to the OPNs theory or method you use.

### Foundational OPNs theory

- **OPNs theory:** Lei Zhou, "Ordered pair of normalized real numbers," *Information Sciences*, 538 (2020), 290-313. DOI: [10.1016/j.ins.2020.05.036](https://doi.org/10.1016/j.ins.2020.05.036)
- **OPNs matrix / spectral theory:** Lei Zhou, "Smith Normal Forms and Matrix Theory over Ordered Pair of Normalized Real Numbers," 2026, preprint. DOI: [10.20944/preprints202606.1206.v1](https://doi.org/10.20944/preprints202606.1206.v1)

### Published OPNs learning methods

- **OPNs-SR:** Yonglin Huang, Yi Zheng, Xiaoqin Pan, and Lei Zhou, "Stepwise regression algorithm based on the ordered pair of normalized real numbers framework," *The Journal of Supercomputing*, 81, Article 900, 2025. DOI: [10.1007/s11227-025-07369-6](https://doi.org/10.1007/s11227-025-07369-6)
- **OPNs-kNN:** Yi Zheng, Xuanbin Ding, Xiang Zhao, Xiaoqin Pan, and Lei Zhou, "K-Nearest Neighbor Algorithm Based on the Framework of Ordered Pair of Normalized Real Numbers," *IEEE Transactions on Artificial Intelligence*, 6(11), 3132-3147, 2025. DOI: [10.1109/TAI.2025.3566925](https://doi.org/10.1109/TAI.2025.3566925)
- **OPNs-K-means:** Ying Tang, Jia Guo, Yi Zheng, Hao Feng, Xiaoqin Pan, and Lei Zhou, "K-means clustering with generalized metrics using Ordered Pair of Normalized real numbers," *Pattern Recognition*, 180, Article 114236, 2026. DOI: [10.1016/j.patcog.2026.114236](https://doi.org/10.1016/j.patcog.2026.114236)

### Repository research projects

**OPNs-LR** and **OPNs-HybridBoost** are active research projects hosted in this repository. Their manuscript and reproducibility materials are provided as research artifacts, but they are not listed above as formally published scholarly references.

- **OPNs-LR:** see [`OPNs-LR.pdf`](./OPNs-LR.pdf) and the root OPNs-LR implementation.
- **OPNs-HybridBoost:** see the [`research/hybridboost/`](./research/hybridboost/) paper-code and reproducibility package.

Formal publication citations for these projects will be added when stable publication records become available.

Repository-level software citation metadata is provided in [`CITATION.cff`](./CITATION.cff). Because this repository supports multiple OPNs research projects, the repository-level software citation does not replace the method-specific scholarly citations above.
