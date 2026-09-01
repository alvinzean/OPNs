from __future__ import annotations

"""Reproduce the final OPNs-HybridBoost warm-start ablation.

Formal comparison
-----------------
Classification:
    datasets = breast_cancer, wine, car, iris
    modes = full_all, tree_only_all
    five folds, one repeat, random_state=161803
    checkpoint_step=10
    round budget capped at 60

Regression:
    datasets = airfoil, concrete, energy_cooling, wine_quality
    modes = full_all, tree_only_all
    five folds, one repeat, random_state=161803
    checkpoint_step=10
    Wine Quality uses train-fold-only median imputation.

Each model is fitted once at its final budget. Intermediate learning-curve
points are obtained from the verified staged-prediction helpers rather than
by retraining at every checkpoint.

The fold/checkpoint-level raw_curve.csv files are the primary
reproducibility artifacts.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


REPO_ROOT = (
    Path(__file__).resolve().parents[3]
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


import opns_pack.opns_np as op

from opns_pack.opns import OPNs

from opns_boost.data import (
    load_classification_dataset,
    load_regression_dataset,
    make_opns_features,
    prepare_regression_fold,
    repeated_splits,
)
from opns_boost.experiment import (
    resolve_dataset_params,
)

from research.hybridboost.scripts import (
    _warm_start_classification as cls_helpers,
)
from research.hybridboost.scripts import (
    _warm_start_regression as reg_helpers,
)


try:
    from threadpoolctl import (
        threadpool_limits,
    )
except ImportError:
    threadpool_limits = None


CONFIG_DIR = (
    REPO_ROOT
    / "research"
    / "hybridboost"
    / "configs"
)

DEFAULT_CLASSIFICATION_CONFIG = (
    CONFIG_DIR
    / "classification_final.json"
)

DEFAULT_REGRESSION_CONFIG = (
    CONFIG_DIR
    / "regression_warm_start_final.json"
)


EXPECTED_CONFIG_HASHES = {
    "classification_final.json": (
        "2f64a5e5e210da757e803124e2bd789ff"
        "20adc8ae2652089b5c492dca23c8f79"
    ),
    "regression_warm_start_final.json": (
        "3845f36adafb14396ad171758c38eafa3"
        "8c27c2159d13c8aece3dc54cd8510c7"
    ),
}


DEFAULT_CLASSIFICATION_DATASETS = [
    "breast_cancer",
    "wine",
    "car",
    "iris",
]

DEFAULT_REGRESSION_DATASETS = [
    "airfoil",
    "concrete",
    "energy_cooling",
    "wine_quality",
]

FORMAL_MODES = [
    "full_all",
    "tree_only_all",
]

FORMAL_RANDOM_STATE = 161803

FORMAL_N_SPLITS = 5

FORMAL_N_REPEATS = 1

FORMAL_CHECKPOINT_STEP = 10

FORMAL_CLASSIFICATION_ROUND_BUDGET_CAP = 60

SUMMARY_DDOF = 1


CLASSIFICATION_METRICS = [
    "LogLoss",
    "Accuracy",
    "F1",
    "AUC",
]

REGRESSION_METRICS = [
    "RMSE",
    "MAE",
    "R2",
]


HIGHER_IS_BETTER = {
    "Accuracy",
    "F1",
    "AUC",
    "R2",
}

LOWER_IS_BETTER = {
    "LogLoss",
    "RMSE",
    "MAE",
}


def sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def uses_fold_local_imputation(
    dataset_name: str,
) -> bool:
    return (
        dataset_name
        == "wine_quality"
    )


def to_opns_target(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    return op.array(
        [
            OPNs(
                float(value),
                0.0,
            )
            for value
            in values
        ]
    )


def _thread_context(
    n_jobs: int,
):
    if (
        threadpool_limits
        is None
        or n_jobs < 1
    ):
        return nullcontext()

    return threadpool_limits(
        limits=n_jobs
    )


def resolve_classification_params(
    config: dict[str, Any],
    dataset_name: str,
    round_budget_cap: int | None,
) -> dict[str, Any]:
    params = dict(
        resolve_dataset_params(
            config,
            dataset_name,
        )
    )

    if round_budget_cap is not None:
        current = int(
            params.get(
                "n_estimators",
                round_budget_cap,
            )
        )

        params[
            "n_estimators"
        ] = min(
            current,
            int(
                round_budget_cap
            ),
        )

    return params


def resolve_regression_params(
    config: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    return dict(
        resolve_dataset_params(
            config,
            dataset_name,
        )
    )


def _curve_summary(
    raw: pd.DataFrame,
    checkpoint_column: str,
    metrics: list[str],
) -> pd.DataFrame:
    successful = raw.loc[
        raw[
            "status"
        ]
        == "ok"
    ].copy()

    if successful.empty:
        return pd.DataFrame()

    rows = []

    for (
        dataset,
        mode,
        checkpoint,
    ), group in successful.groupby(
        [
            "dataset",
            "mode",
            checkpoint_column,
        ],
        sort=False,
    ):
        row = {
            "dataset": dataset,
            "mode": mode,
            checkpoint_column: int(
                checkpoint
            ),
            "folds": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            metrics
            + [
                "time_to_checkpoint",
            ]
        ):
            if metric not in group:
                continue

            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=SUMMARY_DDOF
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def _paired_curve(
    raw: pd.DataFrame,
    checkpoint_column: str,
    metrics: list[str],
) -> pd.DataFrame:
    successful = raw.loc[
        raw[
            "status"
        ]
        == "ok"
    ].copy()

    full = successful.loc[
        successful[
            "mode"
        ]
        == "full_all"
    ].copy()

    tree = successful.loc[
        successful[
            "mode"
        ]
        == "tree_only_all"
    ].copy()

    if (
        full.empty
        or tree.empty
    ):
        return pd.DataFrame()

    keys = [
        "dataset",
        "split",
        checkpoint_column,
    ]

    value_columns = [
        metric
        for metric in metrics
        if metric in successful
    ]

    if (
        "time_to_checkpoint"
        in successful
    ):
        value_columns.append(
            "time_to_checkpoint"
        )

    full = full[
        keys
        + value_columns
    ].rename(
        columns={
            column: (
                f"full_all_{column}"
            )
            for column in value_columns
        }
    )

    tree = tree[
        keys
        + value_columns
    ].rename(
        columns={
            column: (
                f"tree_only_all_{column}"
            )
            for column in value_columns
        }
    )

    paired = full.merge(
        tree,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    for metric in metrics:
        full_column = (
            f"full_all_{metric}"
        )

        tree_column = (
            f"tree_only_all_{metric}"
        )

        if (
            full_column
            not in paired
            or tree_column
            not in paired
        ):
            continue

        if metric in HIGHER_IS_BETTER:
            paired[
                f"gain_{metric}"
            ] = (
                paired[
                    full_column
                ]
                - paired[
                    tree_column
                ]
            )

        elif metric in LOWER_IS_BETTER:
            paired[
                f"gain_{metric}"
            ] = (
                paired[
                    tree_column
                ]
                - paired[
                    full_column
                ]
            )

    if (
        "full_all_time_to_checkpoint"
        in paired
        and "tree_only_all_time_to_checkpoint"
        in paired
    ):
        paired[
            "extra_time_full"
        ] = (
            paired[
                "full_all_time_to_checkpoint"
            ]
            - paired[
                "tree_only_all_time_to_checkpoint"
            ]
        )

    return paired


def _paired_summary(
    paired: pd.DataFrame,
    checkpoint_column: str,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    value_columns = [
        column
        for column in paired.columns
        if (
            column.startswith(
                "gain_"
            )
            or column
            == "extra_time_full"
        )
    ]

    rows = []

    for (
        dataset,
        checkpoint,
    ), group in paired.groupby(
        [
            "dataset",
            checkpoint_column,
        ],
        sort=False,
    ):
        row = {
            "dataset": dataset,
            checkpoint_column: int(
                checkpoint
            ),
            "folds": int(
                len(
                    group
                )
            ),
        }

        for column in value_columns:
            values = pd.to_numeric(
                group[
                    column
                ],
                errors="coerce",
            )

            row[
                f"{column}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{column}_std"
            ] = float(
                values.std(
                    ddof=SUMMARY_DDOF
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def _write_task_outputs(
    task_dir: Path,
    raw: pd.DataFrame,
    checkpoint_column: str,
    metrics: list[str],
) -> None:
    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_summary = (
        _curve_summary(
            raw,
            checkpoint_column,
            metrics,
        )
    )

    paired = (
        _paired_curve(
            raw,
            checkpoint_column,
            metrics,
        )
    )

    paired_summary = (
        _paired_summary(
            paired,
            checkpoint_column,
        )
    )

    raw.to_csv(
        task_dir
        / "raw_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    curve_summary.to_csv(
        task_dir
        / "curve_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        task_dir
        / "paired_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_summary.to_csv(
        task_dir
        / "paired_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_classification(
    args: argparse.Namespace,
) -> pd.DataFrame:
    config = load_json(
        args.classification_config
    )

    records = []

    for dataset_name in (
        args.classification_datasets
    ):
        dataset = (
            load_classification_dataset(
                dataset_name,
                data_root=args.data_root,
            )
        )

        prep_started = (
            time.perf_counter()
        )

        X_opns, pair_features = (
            make_opns_features(
                dataset.X,
                mode="combinations",
            )
        )

        candidate_pairs = (
            len(
                pair_features
            )
            // 2
        )

        preparation_time = (
            time.perf_counter()
            - prep_started
        )

        y = np.asarray(
            dataset.y,
            dtype=int,
        ).reshape(-1)

        params = (
            resolve_classification_params(
                config,
                dataset_name,
                args.classification_round_budget_cap,
            )
        )

        final_budget = int(
            params[
                "n_estimators"
            ]
        )

        checkpoints = set(
            cls_helpers.checkpoint_grid(
                final_budget,
                args.checkpoint_step,
            )
        )

        splits = list(
            repeated_splits(
                y,
                "classification",
                args.n_splits,
                args.n_repeats,
                args.random_state,
            )
        )

        for (
            split_id,
            (
                train_idx,
                test_idx,
            ),
        ) in enumerate(
            splits
        ):
            seed = (
                args.random_state
                + split_id
            )

            X_train = (
                X_opns[
                    train_idx
                ]
            )

            X_test = (
                X_opns[
                    test_idx
                ]
            )

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]

            offset = (
                split_id
                % len(
                    args.modes
                )
            )

            mode_order = (
                list(
                    args.modes[
                        offset:
                    ]
                )
                + list(
                    args.modes[
                        :offset
                    ]
                )
            )

            for (
                order_index,
                mode,
            ) in enumerate(
                mode_order
            ):
                gc.collect()

                fit_started = (
                    time.perf_counter()
                )

                try:
                    with _thread_context(
                        args.n_jobs
                    ):
                        model = (
                            cls_helpers
                            .build_model(
                                mode,
                                params,
                                seed,
                            )
                        )

                        model.fit(
                            X_train,
                            y_train,
                        )

                    fit_wall_time = (
                        time.perf_counter()
                        - fit_started
                    )

                    diagnostics = (
                        model.get_diagnostics()
                    )

                    fitted_budget = int(
                        len(
                            model.estimators_[
                                0
                            ].trees_
                        )
                    )

                    if (
                        fitted_budget
                        != final_budget
                    ):
                        raise RuntimeError(
                            "Classification fitted round "
                            f"budget {fitted_budget} does "
                            f"not match requested "
                            f"{final_budget}."
                        )

                    staged = (
                        cls_helpers
                        .staged_probabilities(
                            model,
                            X_test,
                            checkpoints,
                        )
                    )

                    direct = (
                        model.predict_proba(
                            X_test
                        )
                    )

                    final_staged = (
                        staged[
                            fitted_budget
                        ]
                    )

                    max_abs_diff = float(
                        np.max(
                            np.abs(
                                direct
                                - final_staged
                            )
                        )
                    )

                    if (
                        max_abs_diff
                        > args.equivalence_atol
                    ):
                        raise RuntimeError(
                            "Classification staged/direct "
                            "prediction mismatch: "
                            f"{max_abs_diff:.3e} > "
                            f"{args.equivalence_atol:.3e}"
                        )

                    n_binary = int(
                        len(
                            model.estimators_
                        )
                    )

                    for n_rounds in sorted(
                        checkpoints
                    ):
                        scores = (
                            cls_helpers
                            .classification_scores(
                                y_test,
                                staged[
                                    n_rounds
                                ],
                                model.classes_,
                            )
                        )

                        records.append(
                            {
                                "dataset": (
                                    dataset_name
                                ),
                                "task_type": (
                                    "classification"
                                ),
                                "split": int(
                                    split_id
                                ),
                                "seed": int(
                                    seed
                                ),
                                "mode": mode,
                                "mode_order_index": (
                                    int(
                                        order_index
                                    )
                                ),
                                "mode_execution_order": (
                                    ">".join(
                                        mode_order
                                    )
                                ),
                                "status": "ok",
                                "error": "",
                                "n_samples": int(
                                    len(
                                        y
                                    )
                                ),
                                "n_train": int(
                                    len(
                                        train_idx
                                    )
                                ),
                                "n_test": int(
                                    len(
                                        test_idx
                                    )
                                ),
                                "candidate_pairs": int(
                                    candidate_pairs
                                ),
                                "n_classes": int(
                                    len(
                                        model.classes_
                                    )
                                ),
                                "n_binary_estimators": (
                                    n_binary
                                ),
                                "final_round_budget": (
                                    fitted_budget
                                ),
                                "n_rounds": int(
                                    n_rounds
                                ),
                                "total_trees": int(
                                    n_rounds
                                    * n_binary
                                ),
                                "round_fraction": (
                                    float(
                                        n_rounds
                                        / max(
                                            fitted_budget,
                                            1,
                                        )
                                    )
                                ),
                                "phase1_enabled": (
                                    bool(
                                        diagnostics.get(
                                            "phase1_enabled",
                                            mode
                                            != "tree_only_all",
                                        )
                                    )
                                ),
                                "tree_feature_mode": (
                                    str(
                                        diagnostics.get(
                                            "tree_feature_mode",
                                            "all",
                                        )
                                    )
                                ),
                                "preparation_time": (
                                    float(
                                        preparation_time
                                    )
                                ),
                                "phase1_time": (
                                    float(
                                        diagnostics.get(
                                            "phase1_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "feature_pool_time": (
                                    float(
                                        diagnostics.get(
                                            "feature_pool_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "threshold_cache_time": (
                                    float(
                                        diagnostics.get(
                                            "threshold_cache_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "phase2_time_full": (
                                    float(
                                        diagnostics.get(
                                            "phase2_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "fit_wall_time": (
                                    float(
                                        fit_wall_time
                                    )
                                ),
                                "model_total_fit_time": (
                                    float(
                                        diagnostics.get(
                                            "total_fit_time",
                                            fit_wall_time,
                                        )
                                    )
                                ),
                                "time_to_checkpoint": (
                                    cls_helpers
                                    .time_to_checkpoint(
                                        model,
                                        n_rounds,
                                    )
                                ),
                                "candidate_feature_evaluations_cumulative": (
                                    cls_helpers
                                    .cumulative_sum(
                                        model,
                                        "tree_candidate_feature_evaluations_",
                                        n_rounds,
                                    )
                                ),
                                "threshold_evaluations_cumulative": (
                                    cls_helpers
                                    .cumulative_sum(
                                        model,
                                        "tree_threshold_evaluations_",
                                        n_rounds,
                                    )
                                ),
                                "staged_direct_max_abs_diff": (
                                    max_abs_diff
                                ),
                                **scores,
                            }
                        )

                except Exception as exc:
                    records.append(
                        {
                            "dataset": (
                                dataset_name
                            ),
                            "task_type": (
                                "classification"
                            ),
                            "split": int(
                                split_id
                            ),
                            "seed": int(
                                seed
                            ),
                            "mode": mode,
                            "mode_order_index": (
                                int(
                                    order_index
                                )
                            ),
                            "mode_execution_order": (
                                ">".join(
                                    mode_order
                                )
                            ),
                            "status": "error",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                            "candidate_pairs": int(
                                candidate_pairs
                            ),
                        }
                    )

                    raise

    return pd.DataFrame(
        records
    )


def run_regression(
    args: argparse.Namespace,
) -> pd.DataFrame:
    config = load_json(
        args.regression_config
    )

    records = []

    for dataset_name in (
        args.regression_datasets
    ):
        use_fold_local = (
            uses_fold_local_imputation(
                dataset_name
            )
        )

        dataset_started = (
            time.perf_counter()
        )

        dataset = (
            load_regression_dataset(
                dataset_name,
                data_root=args.data_root,
                defer_imputation=(
                    use_fold_local
                ),
            )
        )

        X_opns = None

        candidate_pairs = None

        if not use_fold_local:
            X_opns, pair_features = (
                make_opns_features(
                    dataset.X,
                    mode="combinations",
                )
            )

            candidate_pairs = (
                len(
                    pair_features
                )
                // 2
            )

        preparation_time = (
            time.perf_counter()
            - dataset_started
        )

        y_all = np.asarray(
            dataset.y,
            dtype=float,
        ).reshape(-1)

        params = (
            resolve_regression_params(
                config,
                dataset_name,
            )
        )

        final_budget = int(
            params[
                "n_estimators"
            ]
        )

        checkpoints = set(
            reg_helpers.checkpoint_grid(
                final_budget,
                args.checkpoint_step,
            )
        )

        splits = list(
            repeated_splits(
                y_all,
                "regression",
                args.n_splits,
                args.n_repeats,
                args.random_state,
            )
        )

        for (
            split_id,
            (
                train_idx,
                test_idx,
            ),
        ) in enumerate(
            splits
        ):
            seed = (
                args.random_state
                + split_id
            )

            fold_preparation_started = (
                time.perf_counter()
            )

            if use_fold_local:
                prepared = (
                    prepare_regression_fold(
                        dataset,
                        train_idx,
                        test_idx,
                        need_opns=True,
                        fold_local_imputation=True,
                    )
                )

                X_train = (
                    prepared.X_train_opns
                )

                X_test = (
                    prepared.X_test_opns
                )

                y_train_raw = (
                    np.asarray(
                        prepared.y_train,
                        dtype=float,
                    ).reshape(-1)
                )

                y_test_raw = (
                    np.asarray(
                        prepared.y_test,
                        dtype=float,
                    ).reshape(-1)
                )

                fold_candidate_pairs = (
                    len(
                        prepared.pair_features
                    )
                    // 2
                )

            else:
                X_train = (
                    X_opns[
                        train_idx
                    ]
                )

                X_test = (
                    X_opns[
                        test_idx
                    ]
                )

                y_train_raw = (
                    y_all[
                        train_idx
                    ]
                )

                y_test_raw = (
                    y_all[
                        test_idx
                    ]
                )

                fold_candidate_pairs = int(
                    candidate_pairs
                )

            fold_preparation_time = (
                time.perf_counter()
                - fold_preparation_started
            )

            target_scaler = (
                MinMaxScaler()
                .fit(
                    y_train_raw.reshape(
                        -1,
                        1,
                    )
                )
            )

            y_train_scaled = (
                target_scaler
                .transform(
                    y_train_raw.reshape(
                        -1,
                        1,
                    )
                )
                .reshape(-1)
            )

            y_train_model = (
                to_opns_target(
                    y_train_scaled
                )
            )

            offset = (
                split_id
                % len(
                    args.modes
                )
            )

            mode_order = (
                list(
                    args.modes[
                        offset:
                    ]
                )
                + list(
                    args.modes[
                        :offset
                    ]
                )
            )

            for (
                order_index,
                mode,
            ) in enumerate(
                mode_order
            ):
                gc.collect()

                fit_started = (
                    time.perf_counter()
                )

                try:
                    with _thread_context(
                        args.n_jobs
                    ):
                        model = (
                            reg_helpers
                            .build_model(
                                mode,
                                params,
                                seed,
                            )
                        )

                        model.fit(
                            X_train,
                            y_train_model,
                        )

                    fit_wall_time = (
                        time.perf_counter()
                        - fit_started
                    )

                    diagnostics = (
                        model.get_diagnostics()
                    )

                    fitted_budget = int(
                        len(
                            model.trees_
                        )
                    )

                    if (
                        fitted_budget
                        != final_budget
                    ):
                        raise RuntimeError(
                            "Regression fitted tree "
                            f"budget {fitted_budget} does "
                            f"not match requested "
                            f"{final_budget}."
                        )

                    predictions = (
                        reg_helpers
                        .staged_predictions(
                            model,
                            X_test,
                            checkpoints,
                        )
                    )

                    final_direct = (
                        model.predict(
                            X_test,
                            item=1,
                            backend="object_batch",
                        )
                    )

                    final_staged = (
                        predictions[
                            fitted_budget
                        ]
                    )

                    max_abs_diff = float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    final_direct,
                                    dtype=float,
                                ).reshape(-1)
                                - final_staged
                            )
                        )
                    )

                    if (
                        max_abs_diff
                        > args.equivalence_atol
                    ):
                        raise RuntimeError(
                            "Regression staged/direct "
                            "prediction mismatch: "
                            f"{max_abs_diff:.3e} > "
                            f"{args.equivalence_atol:.3e}"
                        )

                    selected = float(
                        diagnostics.get(
                            "tree_search_features",
                            fold_candidate_pairs,
                        )
                    )

                    retention = (
                        selected
                        / max(
                            fold_candidate_pairs,
                            1,
                        )
                    )

                    for n_trees in sorted(
                        checkpoints
                    ):
                        pred_raw = (
                            target_scaler
                            .inverse_transform(
                                predictions[
                                    n_trees
                                ].reshape(
                                    -1,
                                    1,
                                )
                            )
                            .reshape(-1)
                        )

                        scores = (
                            reg_helpers
                            .regression_scores(
                                y_test_raw,
                                pred_raw,
                            )
                        )

                        records.append(
                            {
                                "dataset": (
                                    dataset_name
                                ),
                                "task_type": (
                                    "regression"
                                ),
                                "split": int(
                                    split_id
                                ),
                                "seed": int(
                                    seed
                                ),
                                "mode": mode,
                                "mode_order_index": (
                                    int(
                                        order_index
                                    )
                                ),
                                "mode_execution_order": (
                                    ">".join(
                                        mode_order
                                    )
                                ),
                                "status": "ok",
                                "error": "",
                                "n_samples": int(
                                    len(
                                        y_all
                                    )
                                ),
                                "n_train": int(
                                    len(
                                        train_idx
                                    )
                                ),
                                "n_test": int(
                                    len(
                                        test_idx
                                    )
                                ),
                                "candidate_pairs": int(
                                    fold_candidate_pairs
                                ),
                                "fold_local_imputation": (
                                    bool(
                                        use_fold_local
                                    )
                                ),
                                "final_tree_budget": (
                                    fitted_budget
                                ),
                                "n_trees": int(
                                    n_trees
                                ),
                                "tree_fraction": (
                                    float(
                                        n_trees
                                        / max(
                                            fitted_budget,
                                            1,
                                        )
                                    )
                                ),
                                "phase1_enabled": (
                                    bool(
                                        diagnostics.get(
                                            "phase1_enabled",
                                            mode
                                            != "tree_only_all",
                                        )
                                    )
                                ),
                                "tree_feature_mode": (
                                    str(
                                        diagnostics.get(
                                            "tree_feature_mode",
                                            "all",
                                        )
                                    )
                                ),
                                "tree_search_features": (
                                    selected
                                ),
                                "active_retention_ratio": (
                                    float(
                                        retention
                                    )
                                ),
                                "preparation_time": (
                                    float(
                                        preparation_time
                                    )
                                ),
                                "fold_preparation_time": (
                                    float(
                                        fold_preparation_time
                                    )
                                ),
                                "phase1_time": (
                                    float(
                                        diagnostics.get(
                                            "phase1_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "feature_pool_time": (
                                    float(
                                        diagnostics.get(
                                            "feature_pool_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "threshold_cache_time": (
                                    float(
                                        diagnostics.get(
                                            "threshold_cache_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "phase2_time_full": (
                                    float(
                                        diagnostics.get(
                                            "phase2_time",
                                            0.0,
                                        )
                                    )
                                ),
                                "fit_wall_time": (
                                    float(
                                        fit_wall_time
                                    )
                                ),
                                "model_total_fit_time": (
                                    float(
                                        diagnostics.get(
                                            "total_fit_time",
                                            fit_wall_time,
                                        )
                                    )
                                ),
                                "time_to_checkpoint": (
                                    reg_helpers
                                    .time_to_checkpoint(
                                        model,
                                        n_trees,
                                    )
                                ),
                                "candidate_feature_evaluations_cumulative": (
                                    reg_helpers
                                    .cumulative_count(
                                        model,
                                        "tree_candidate_feature_evaluations_",
                                        n_trees,
                                    )
                                ),
                                "threshold_evaluations_cumulative": (
                                    reg_helpers
                                    .cumulative_count(
                                        model,
                                        "tree_threshold_evaluations_",
                                        n_trees,
                                    )
                                ),
                                "staged_direct_max_abs_diff": (
                                    max_abs_diff
                                ),
                                **scores,
                            }
                        )

                except Exception as exc:
                    records.append(
                        {
                            "dataset": (
                                dataset_name
                            ),
                            "task_type": (
                                "regression"
                            ),
                            "split": int(
                                split_id
                            ),
                            "seed": int(
                                seed
                            ),
                            "mode": mode,
                            "mode_order_index": (
                                int(
                                    order_index
                                )
                            ),
                            "mode_execution_order": (
                                ">".join(
                                    mode_order
                                )
                            ),
                            "status": "error",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                            "candidate_pairs": int(
                                fold_candidate_pairs
                            ),
                            "fold_local_imputation": (
                                bool(
                                    use_fold_local
                                )
                            ),
                        }
                    )

                    raise

    return pd.DataFrame(
        records
    )


def package_version(
    package: str,
):
    try:
        return importlib.metadata.version(
            package
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def build_protocol(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "study": (
            "OPNs-HybridBoost"
        ),
        "experiment": (
            "warm_start_ablation"
        ),
        "task": args.task,
        "classification_datasets": (
            args.classification_datasets
        ),
        "regression_datasets": (
            args.regression_datasets
        ),
        "modes": args.modes,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "random_state": (
            args.random_state
        ),
        "checkpoint_step": (
            args.checkpoint_step
        ),
        "classification_round_budget_cap": (
            args.classification_round_budget_cap
        ),
        "summary_std": {
            "type": (
                "sample_standard_deviation"
            ),
            "ddof": (
                SUMMARY_DDOF
            ),
        },
        "classification": {
            "full_all": {
                "phase1_enabled": True,
                "tree_feature_mode": (
                    "all"
                ),
            },
            "tree_only_all": {
                "phase1_enabled": False,
                "tree_feature_mode": (
                    "all"
                ),
            },
            "staging": (
                "single fitted model; exact "
                "prefix probabilities"
            ),
        },
        "regression": {
            "full_all": {
                "model": (
                    "OPNsHybridRegressor"
                ),
                "tree_feature_mode": (
                    "all"
                ),
            },
            "tree_only_all": {
                "model": (
                    "OPNsTreeOnlyRegressor"
                ),
                "warm_start_mode": (
                    "constant"
                ),
                "tree_feature_mode": (
                    "all"
                ),
            },
            "prediction_backend": (
                "object_batch"
            ),
            "target_scaling": (
                "MinMaxScaler fitted on "
                "training targets only"
            ),
            "wine_quality_imputation": (
                "training-fold median only"
            ),
            "staging": (
                "single fitted model; exact "
                "prefix predictions"
            ),
        },
        "configs": {
            "classification": {
                "path": str(
                    args.classification_config
                ),
                "sha256": sha256(
                    args.classification_config
                ),
            },
            "regression": {
                "path": str(
                    args.regression_config
                ),
                "sha256": sha256(
                    args.regression_config
                ),
            },
        },
        "raw_curve_is_primary": (
            True
        ),
    }


def write_metadata(
    args: argparse.Namespace,
) -> None:
    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        args.out_dir
        / "protocol.json"
    ).write_text(
        json.dumps(
            build_protocol(
                args
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    environment = {
        "python": sys.version,
        "platform": (
            platform.platform()
        ),
        "processor": (
            platform.processor()
        ),
        "cpu_count": (
            os.cpu_count()
        ),
        "packages": {
            "numpy": (
                package_version(
                    "numpy"
                )
            ),
            "pandas": (
                package_version(
                    "pandas"
                )
            ),
            "scikit-learn": (
                package_version(
                    "scikit-learn"
                )
            ),
        },
        "arguments": {
            key: (
                str(
                    value
                )
                if isinstance(
                    value,
                    Path,
                )
                else value
            )
            for key, value
            in vars(
                args
            ).items()
        },
    }

    (
        args.out_dir
        / "environment.json"
    ).write_text(
        json.dumps(
            environment,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for source in (
        args.classification_config,
        args.regression_config,
    ):
        shutil.copyfile(
            source,
            args.out_dir
            / source.name,
        )


def parse_args(
    argv=None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the final "
            "OPNs-HybridBoost warm-start "
            "ablation."
        )
    )

    parser.add_argument(
        "--task",
        choices=[
            "all",
            "classification",
            "regression",
        ],
        default="all",
    )

    parser.add_argument(
        "--classification-datasets",
        nargs="+",
        default=list(
            DEFAULT_CLASSIFICATION_DATASETS
        ),
    )

    parser.add_argument(
        "--regression-datasets",
        nargs="+",
        default=list(
            DEFAULT_REGRESSION_DATASETS
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FORMAL_MODES,
        default=list(
            FORMAL_MODES
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=FORMAL_N_SPLITS,
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=FORMAL_N_REPEATS,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=FORMAL_RANDOM_STATE,
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=(
            FORMAL_CHECKPOINT_STEP
        ),
    )

    parser.add_argument(
        "--classification-round-budget-cap",
        type=int,
        default=(
            FORMAL_CLASSIFICATION_ROUND_BUDGET_CAP
        ),
    )

    parser.add_argument(
        "--equivalence-atol",
        type=float,
        default=1.0e-10,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--classification-config",
        type=Path,
        default=(
            DEFAULT_CLASSIFICATION_CONFIG
        ),
    )

    parser.add_argument(
        "--regression-config",
        type=Path,
        default=(
            DEFAULT_REGRESSION_CONFIG
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "research"
            / "hybridboost"
            / "results"
            / "reproduced_warm_start"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the formal protocol/configs "
            "and write metadata without fitting."
        ),
    )

    args = parser.parse_args(
        argv
    )

    if args.n_splits < 2:
        parser.error(
            "--n-splits must be at least 2"
        )

    if args.n_repeats < 1:
        parser.error(
            "--n-repeats must be at least 1"
        )

    if args.checkpoint_step < 1:
        parser.error(
            "--checkpoint-step must be positive"
        )

    if (
        args.classification_round_budget_cap
        < 1
    ):
        parser.error(
            "--classification-round-budget-cap "
            "must be positive"
        )

    if args.equivalence_atol < 0:
        parser.error(
            "--equivalence-atol cannot be negative"
        )

    if args.n_jobs == 0:
        parser.error(
            "--n-jobs cannot be 0"
        )

    if len(
        set(
            args.modes
        )
    ) != len(
        args.modes
    ):
        parser.error(
            "--modes cannot contain duplicates"
        )

    for path in (
        args.classification_config,
        args.regression_config,
    ):
        if not path.is_file():
            parser.error(
                f"Config does not exist: {path}"
            )

    return args


def main(
    argv=None,
) -> int:
    args = parse_args(
        argv
    )

    write_metadata(
        args
    )

    if args.dry_run:
        print(
            json.dumps(
                build_protocol(
                    args
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "\nDry-run metadata written to: "
            f"{args.out_dir.resolve()}"
        )

        return 0

    if args.task in {
        "all",
        "classification",
    }:
        raw = run_classification(
            args
        )

        _write_task_outputs(
            args.out_dir
            / "classification",
            raw,
            "n_rounds",
            CLASSIFICATION_METRICS,
        )

    if args.task in {
        "all",
        "regression",
    }:
        raw = run_regression(
            args
        )

        _write_task_outputs(
            args.out_dir
            / "regression",
            raw,
            "n_trees",
            REGRESSION_METRICS,
        )

    print(
        "\nWarm-start results written to: "
        f"{args.out_dir.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )