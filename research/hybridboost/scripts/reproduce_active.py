from __future__ import annotations

"""Reproduce the final OPNs-HybridBoost active-feature studies.

Experiments
-----------
efficiency (default):
    regression only
    datasets = energy_cooling, wine_quality
    modes = full_all, full_active
    five folds, one repeat, random_state=161803
    checkpoint_step=10
    config = regression_warm_start_final.json

    Each mode is fitted once at its final tree budget. Intermediate
    checkpoints are obtained from the frozen staged-prediction helper.
    Pairwise active-vs-all diagnostics are generated directly from
    raw_curve.csv.

structure:
    protocol name = phase1_structure_snapshot
    regression only
    datasets = airfoil, concrete, energy_cooling, wine_quality
    mode = full_active
    five folds, one repeat, random_state=161803
    tree budget forced to zero

    This is intentionally a Phase-I active-set structure snapshot, not a
    boosting learning curve.
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


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import opns_pack.opns_np as op

from opns_pack.opns import OPNs

from opns_boost.data import (
    load_regression_dataset,
    make_opns_features,
    prepare_regression_fold,
    repeated_splits,
)
from opns_boost.experiment import resolve_dataset_params

from research.hybridboost.scripts import _warm_start_regression as reg_helpers


try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


CONFIG_DIR = REPO_ROOT / "research" / "hybridboost" / "configs"

DEFAULT_REGRESSION_CONFIG = CONFIG_DIR / "regression_warm_start_final.json"

EXPECTED_REGRESSION_CONFIG_HASH = (
    "3845f36adafb14396ad171758c38eafa3"
    "8c27c2159d13c8aece3dc54cd8510c7"
)

FORMAL_EFFICIENCY_DATASETS = [
    "energy_cooling",
    "wine_quality",
]

FORMAL_STRUCTURE_DATASETS = [
    "airfoil",
    "concrete",
    "energy_cooling",
    "wine_quality",
]

FORMAL_EFFICIENCY_MODES = [
    "full_all",
    "full_active",
]

FORMAL_STRUCTURE_MODE = "full_active"

FORMAL_RANDOM_STATE = 161803

FORMAL_N_SPLITS = 5

FORMAL_N_REPEATS = 1

FORMAL_CHECKPOINT_STEP = 10

FORMAL_STRUCTURE_TREE_BUDGET_CAP = 0

SUMMARY_DDOF = 1

REGRESSION_METRICS = [
    "RMSE",
    "MAE",
    "R2",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return value


def uses_fold_local_imputation(dataset_name: str) -> bool:
    return dataset_name == "wine_quality"


def to_opns_target(values):
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
            for value in values
        ]
    )


def _thread_context(n_jobs: int):
    if threadpool_limits is None or n_jobs < 1:
        return nullcontext()

    return threadpool_limits(
        limits=n_jobs,
    )


def resolve_regression_params(
    config: dict[str, Any],
    dataset_name: str,
    tree_budget_cap: int | None = None,
) -> dict[str, Any]:
    params = dict(
        resolve_dataset_params(
            config,
            dataset_name,
        )
    )

    if tree_budget_cap is not None:
        current = int(
            params.get(
                "n_estimators",
                tree_budget_cap,
            )
        )
        params["n_estimators"] = min(
            current,
            int(tree_budget_cap),
        )

    return params


def _curve_summary(raw: pd.DataFrame) -> pd.DataFrame:
    successful = raw.loc[
        raw["status"] == "ok"
    ].copy()

    if successful.empty:
        return pd.DataFrame()

    rows = []

    for (
        dataset,
        mode,
        n_trees,
    ), group in successful.groupby(
        [
            "dataset",
            "mode",
            "n_trees",
        ],
        sort=False,
    ):
        row = {
            "dataset": dataset,
            "mode": mode,
            "n_trees": int(n_trees),
            "folds": int(len(group)),
        }

        for metric in (
            REGRESSION_METRICS
            + [
                "time_to_checkpoint",
                "active_retention_ratio",
                "candidate_feature_evaluations_cumulative",
                "threshold_evaluations_cumulative",
            ]
        ):
            if metric not in group:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = float(
                values.mean()
            )
            row[f"{metric}_std"] = float(
                values.std(
                    ddof=SUMMARY_DDOF,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype(float)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype(float)

    values = np.divide(
        numerator.to_numpy(),
        denominator.to_numpy(),
        out=np.full(
            len(numerator),
            np.nan,
            dtype=float,
        ),
        where=(
            denominator.to_numpy()
            != 0.0
        ),
    )

    return pd.Series(
        values,
        index=numerator.index,
        dtype=float,
    )


def _active_pairwise_curve(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    successful = raw.loc[
        raw["status"] == "ok"
    ].copy()

    full_all = successful.loc[
        successful["mode"] == "full_all"
    ].copy()

    full_active = successful.loc[
        successful["mode"] == "full_active"
    ].copy()

    if full_all.empty or full_active.empty:
        return pd.DataFrame()

    keys = [
        "dataset",
        "split",
        "n_trees",
    ]

    value_columns = [
        "RMSE",
        "MAE",
        "R2",
        "time_to_checkpoint",
        "tree_search_features",
        "active_retention_ratio",
        "candidate_feature_evaluations_cumulative",
        "threshold_evaluations_cumulative",
    ]

    value_columns = [
        column
        for column in value_columns
        if column in successful
    ]

    full_all = full_all[
        keys + value_columns
    ].rename(
        columns={
            column: f"full_all_{column}"
            for column in value_columns
        }
    )

    full_active = full_active[
        keys + value_columns
    ].rename(
        columns={
            column: f"full_active_{column}"
            for column in value_columns
        }
    )

    paired = full_all.merge(
        full_active,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    for metric in REGRESSION_METRICS:
        all_column = f"full_all_{metric}"
        active_column = f"full_active_{metric}"

        if (
            all_column in paired
            and active_column in paired
        ):
            paired[
                f"{metric}_delta_active_minus_all"
            ] = (
                paired[active_column]
                - paired[all_column]
            )

    if "full_active_active_retention_ratio" in paired:
        paired["active_retention"] = paired[
            "full_active_active_retention_ratio"
        ]

    if (
        "full_active_candidate_feature_evaluations_cumulative"
        in paired
        and "full_all_candidate_feature_evaluations_cumulative"
        in paired
    ):
        paired["candidate_eval_ratio"] = _safe_ratio(
            paired[
                "full_active_candidate_feature_evaluations_cumulative"
            ],
            paired[
                "full_all_candidate_feature_evaluations_cumulative"
            ],
        )

    if (
        "full_active_threshold_evaluations_cumulative"
        in paired
        and "full_all_threshold_evaluations_cumulative"
        in paired
    ):
        paired["threshold_eval_ratio"] = _safe_ratio(
            paired[
                "full_active_threshold_evaluations_cumulative"
            ],
            paired[
                "full_all_threshold_evaluations_cumulative"
            ],
        )

    if (
        "full_active_time_to_checkpoint"
        in paired
        and "full_all_time_to_checkpoint"
        in paired
    ):
        paired["time_ratio"] = _safe_ratio(
            paired[
                "full_active_time_to_checkpoint"
            ],
            paired[
                "full_all_time_to_checkpoint"
            ],
        )

    return paired


def _active_pairwise_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    value_columns = [
        "full_all_RMSE",
        "full_active_RMSE",
        "RMSE_delta_active_minus_all",
        "full_all_MAE",
        "full_active_MAE",
        "MAE_delta_active_minus_all",
        "full_all_R2",
        "full_active_R2",
        "R2_delta_active_minus_all",
        "full_all_time_to_checkpoint",
        "full_active_time_to_checkpoint",
        "full_all_tree_search_features",
        "full_active_tree_search_features",
        "active_retention",
        "candidate_eval_ratio",
        "threshold_eval_ratio",
        "time_ratio",
    ]

    value_columns = [
        column
        for column in value_columns
        if column in paired
    ]

    rows = []

    for (
        dataset,
        n_trees,
    ), group in paired.groupby(
        [
            "dataset",
            "n_trees",
        ],
        sort=False,
    ):
        row = {
            "dataset": dataset,
            "n_trees": int(n_trees),
            "folds": int(len(group)),
        }

        for column in value_columns:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            row[f"{column}_mean"] = float(
                values.mean()
            )
            row[f"{column}_std"] = float(
                values.std(
                    ddof=SUMMARY_DDOF,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _structure_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    successful = raw.loc[
        raw["status"] == "ok"
    ].copy()

    if successful.empty:
        return pd.DataFrame()

    value_columns = [
        "candidate_pairs",
        "tree_search_features",
        "active_retention_ratio",
        "phase1_time",
        "fit_wall_time",
    ]

    rows = []

    for dataset, group in successful.groupby(
        "dataset",
        sort=False,
    ):
        row = {
            "dataset": dataset,
            "folds": int(len(group)),
        }

        for column in value_columns:
            if column not in group:
                continue

            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            row[f"{column}_mean"] = float(
                values.mean()
            )
            row[f"{column}_std"] = float(
                values.std(
                    ddof=SUMMARY_DDOF,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _write_efficiency_outputs(
    out_dir: Path,
    raw: pd.DataFrame,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_summary = _curve_summary(raw)
    paired = _active_pairwise_curve(raw)
    paired_summary = _active_pairwise_summary(
        paired
    )

    raw.to_csv(
        out_dir / "raw_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    curve_summary.to_csv(
        out_dir / "curve_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        out_dir / "active_pairwise_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_summary.to_csv(
        out_dir / "active_pairwise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_structure_outputs(
    out_dir: Path,
    raw: pd.DataFrame,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = _structure_summary(raw)

    raw.to_csv(
        out_dir / "phase1_structure_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        out_dir / "phase1_structure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _dataset_context(
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    use_fold_local = uses_fold_local_imputation(
        dataset_name
    )

    started = time.perf_counter()

    dataset = load_regression_dataset(
        dataset_name,
        data_root=args.data_root,
        defer_imputation=use_fold_local,
    )

    X_opns = None
    candidate_pairs = None

    if not use_fold_local:
        X_opns, pair_features = make_opns_features(
            dataset.X,
            mode="combinations",
        )
        candidate_pairs = (
            len(pair_features)
            // 2
        )

    preparation_time = (
        time.perf_counter()
        - started
    )

    return {
        "dataset": dataset,
        "use_fold_local": use_fold_local,
        "X_opns": X_opns,
        "candidate_pairs": candidate_pairs,
        "preparation_time": preparation_time,
        "y_all": np.asarray(
            dataset.y,
            dtype=float,
        ).reshape(-1),
    }


def _fold_data(
    context: dict[str, Any],
    train_idx,
    test_idx,
) -> dict[str, Any]:
    started = time.perf_counter()

    if context["use_fold_local"]:
        prepared = prepare_regression_fold(
            context["dataset"],
            train_idx,
            test_idx,
            need_opns=True,
            fold_local_imputation=True,
        )

        X_train = prepared.X_train_opns
        X_test = prepared.X_test_opns

        y_train_raw = np.asarray(
            prepared.y_train,
            dtype=float,
        ).reshape(-1)

        y_test_raw = np.asarray(
            prepared.y_test,
            dtype=float,
        ).reshape(-1)

        candidate_pairs = (
            len(prepared.pair_features)
            // 2
        )

    else:
        X_train = context["X_opns"][
            train_idx
        ]
        X_test = context["X_opns"][
            test_idx
        ]

        y_train_raw = context["y_all"][
            train_idx
        ]
        y_test_raw = context["y_all"][
            test_idx
        ]

        candidate_pairs = int(
            context["candidate_pairs"]
        )

    fold_preparation_time = (
        time.perf_counter()
        - started
    )

    target_scaler = MinMaxScaler().fit(
        y_train_raw.reshape(
            -1,
            1,
        )
    )

    y_train_scaled = (
        target_scaler.transform(
            y_train_raw.reshape(
                -1,
                1,
            )
        )
        .reshape(-1)
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_model": to_opns_target(
            y_train_scaled
        ),
        "y_test_raw": y_test_raw,
        "target_scaler": target_scaler,
        "candidate_pairs": candidate_pairs,
        "fold_preparation_time": fold_preparation_time,
    }


def _splits_for_dataset(
    y_all: np.ndarray,
    args: argparse.Namespace,
):
    splits = list(
        repeated_splits(
            y_all,
            "regression",
            args.n_splits,
            args.n_repeats,
            args.random_state,
        )
    )

    if args.max_folds is not None:
        splits = splits[
            : args.max_folds
        ]

    return splits


def run_efficiency(
    args: argparse.Namespace,
) -> pd.DataFrame:
    config = load_json(
        args.regression_config
    )

    records = []

    for dataset_name in args.datasets:
        context = _dataset_context(
            dataset_name,
            args,
        )

        params = resolve_regression_params(
            config,
            dataset_name,
            args.tree_budget_cap,
        )

        final_budget = int(
            params["n_estimators"]
        )

        checkpoints = set(
            reg_helpers.checkpoint_grid(
                final_budget,
                args.checkpoint_step,
            )
        )

        splits = _splits_for_dataset(
            context["y_all"],
            args,
        )

        for split_id, (
            train_idx,
            test_idx,
        ) in enumerate(splits):
            seed = (
                args.random_state
                + split_id
            )

            fold = _fold_data(
                context,
                train_idx,
                test_idx,
            )

            offset = (
                split_id
                % len(
                    FORMAL_EFFICIENCY_MODES
                )
            )

            mode_order = (
                FORMAL_EFFICIENCY_MODES[
                    offset:
                ]
                + FORMAL_EFFICIENCY_MODES[
                    :offset
                ]
            )

            for order_index, mode in enumerate(
                mode_order
            ):
                gc.collect()

                fit_started = time.perf_counter()

                try:
                    with _thread_context(
                        args.n_jobs
                    ):
                        model = reg_helpers.build_model(
                            mode,
                            params,
                            seed,
                        )

                        model.fit(
                            fold["X_train"],
                            fold["y_train_model"],
                        )

                    fit_wall_time = (
                        time.perf_counter()
                        - fit_started
                    )

                    diagnostics = (
                        model.get_diagnostics()
                    )

                    fitted_budget = int(
                        len(model.trees_)
                    )

                    if fitted_budget != final_budget:
                        raise RuntimeError(
                            "Regression fitted tree budget "
                            f"{fitted_budget} does not match "
                            f"requested {final_budget}."
                        )

                    predictions = (
                        reg_helpers.staged_predictions(
                            model,
                            fold["X_test"],
                            checkpoints,
                        )
                    )

                    final_direct = model.predict(
                        fold["X_test"],
                        item=1,
                        backend="object_batch",
                    )

                    final_staged = predictions[
                        fitted_budget
                    ]

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
                            fold["candidate_pairs"],
                        )
                    )

                    retention = (
                        selected
                        / max(
                            fold["candidate_pairs"],
                            1,
                        )
                    )

                    for n_trees in sorted(
                        checkpoints
                    ):
                        pred_raw = (
                            fold["target_scaler"]
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
                            reg_helpers.regression_scores(
                                fold["y_test_raw"],
                                pred_raw,
                            )
                        )

                        records.append(
                            {
                                "dataset": dataset_name,
                                "task_type": "regression",
                                "experiment": "active_efficiency",
                                "split": int(split_id),
                                "seed": int(seed),
                                "mode": mode,
                                "mode_order_index": int(
                                    order_index
                                ),
                                "mode_execution_order": ">".join(
                                    mode_order
                                ),
                                "status": "ok",
                                "error": "",
                                "n_samples": int(
                                    len(
                                        context["y_all"]
                                    )
                                ),
                                "n_train": int(
                                    len(train_idx)
                                ),
                                "n_test": int(
                                    len(test_idx)
                                ),
                                "candidate_pairs": int(
                                    fold[
                                        "candidate_pairs"
                                    ]
                                ),
                                "fold_local_imputation": bool(
                                    context[
                                        "use_fold_local"
                                    ]
                                ),
                                "final_tree_budget": fitted_budget,
                                "n_trees": int(
                                    n_trees
                                ),
                                "tree_fraction": float(
                                    n_trees
                                    / max(
                                        fitted_budget,
                                        1,
                                    )
                                ),
                                "phase1_enabled": bool(
                                    diagnostics.get(
                                        "phase1_enabled",
                                        True,
                                    )
                                ),
                                "tree_feature_mode": str(
                                    diagnostics.get(
                                        "tree_feature_mode",
                                        (
                                            "active"
                                            if mode
                                            == "full_active"
                                            else "all"
                                        ),
                                    )
                                ),
                                "tree_search_features": selected,
                                "active_retention_ratio": float(
                                    retention
                                ),
                                "preparation_time": float(
                                    context[
                                        "preparation_time"
                                    ]
                                ),
                                "fold_preparation_time": float(
                                    fold[
                                        "fold_preparation_time"
                                    ]
                                ),
                                "phase1_time": float(
                                    diagnostics.get(
                                        "phase1_time",
                                        0.0,
                                    )
                                ),
                                "feature_pool_time": float(
                                    diagnostics.get(
                                        "feature_pool_time",
                                        0.0,
                                    )
                                ),
                                "threshold_cache_time": float(
                                    diagnostics.get(
                                        "threshold_cache_time",
                                        0.0,
                                    )
                                ),
                                "phase2_time_full": float(
                                    diagnostics.get(
                                        "phase2_time",
                                        0.0,
                                    )
                                ),
                                "fit_wall_time": float(
                                    fit_wall_time
                                ),
                                "model_total_fit_time": float(
                                    diagnostics.get(
                                        "total_fit_time",
                                        fit_wall_time,
                                    )
                                ),
                                "time_to_checkpoint": (
                                    reg_helpers.time_to_checkpoint(
                                        model,
                                        n_trees,
                                    )
                                ),
                                "candidate_feature_evaluations_cumulative": (
                                    reg_helpers.cumulative_count(
                                        model,
                                        "tree_candidate_feature_evaluations_",
                                        n_trees,
                                    )
                                ),
                                "threshold_evaluations_cumulative": (
                                    reg_helpers.cumulative_count(
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
                            "dataset": dataset_name,
                            "task_type": "regression",
                            "experiment": "active_efficiency",
                            "split": int(split_id),
                            "seed": int(seed),
                            "mode": mode,
                            "mode_order_index": int(
                                order_index
                            ),
                            "mode_execution_order": ">".join(
                                mode_order
                            ),
                            "status": "error",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                            "candidate_pairs": int(
                                fold["candidate_pairs"]
                            ),
                            "fold_local_imputation": bool(
                                context[
                                    "use_fold_local"
                                ]
                            ),
                        }
                    )
                    raise

    return pd.DataFrame(records)


def run_structure(
    args: argparse.Namespace,
) -> pd.DataFrame:
    config = load_json(
        args.regression_config
    )

    records = []

    for dataset_name in args.datasets:
        context = _dataset_context(
            dataset_name,
            args,
        )

        params = resolve_regression_params(
            config,
            dataset_name,
            FORMAL_STRUCTURE_TREE_BUDGET_CAP,
        )

        if int(params["n_estimators"]) != 0:
            raise RuntimeError(
                "Structure protocol must use "
                "tree budget zero."
            )

        splits = _splits_for_dataset(
            context["y_all"],
            args,
        )

        for split_id, (
            train_idx,
            test_idx,
        ) in enumerate(splits):
            seed = (
                args.random_state
                + split_id
            )

            fold = _fold_data(
                context,
                train_idx,
                test_idx,
            )

            gc.collect()

            fit_started = time.perf_counter()

            try:
                with _thread_context(
                    args.n_jobs
                ):
                    model = reg_helpers.build_model(
                        FORMAL_STRUCTURE_MODE,
                        params,
                        seed,
                    )

                    model.fit(
                        fold["X_train"],
                        fold["y_train_model"],
                    )

                fit_wall_time = (
                    time.perf_counter()
                    - fit_started
                )

                diagnostics = (
                    model.get_diagnostics()
                )

                fitted_budget = int(
                    len(model.trees_)
                )

                if fitted_budget != 0:
                    raise RuntimeError(
                        "Phase-I structure snapshot "
                        "must contain zero fitted trees; "
                        f"got {fitted_budget}."
                    )

                phase1_enabled = bool(
                    diagnostics.get(
                        "phase1_enabled",
                        True,
                    )
                )

                tree_feature_mode = str(
                    diagnostics.get(
                        "tree_feature_mode",
                        "active",
                    )
                )

                if not phase1_enabled:
                    raise RuntimeError(
                        "Phase-I structure snapshot "
                        "requires phase1_enabled=True."
                    )

                if tree_feature_mode != "active":
                    raise RuntimeError(
                        "Phase-I structure snapshot "
                        "requires tree_feature_mode="
                        f"'active'; got "
                        f"{tree_feature_mode!r}."
                    )

                selected = float(
                    diagnostics.get(
                        "tree_search_features",
                        fold["candidate_pairs"],
                    )
                )

                retention = (
                    selected
                    / max(
                        fold["candidate_pairs"],
                        1,
                    )
                )

                records.append(
                    {
                        "dataset": dataset_name,
                        "task_type": "regression",
                        "experiment": (
                            "phase1_structure_snapshot"
                        ),
                        "split": int(split_id),
                        "seed": int(seed),
                        "mode": FORMAL_STRUCTURE_MODE,
                        "status": "ok",
                        "error": "",
                        "n_samples": int(
                            len(
                                context["y_all"]
                            )
                        ),
                        "n_train": int(
                            len(train_idx)
                        ),
                        "n_test": int(
                            len(test_idx)
                        ),
                        "candidate_pairs": int(
                            fold["candidate_pairs"]
                        ),
                        "fold_local_imputation": bool(
                            context[
                                "use_fold_local"
                            ]
                        ),
                        "final_tree_budget": 0,
                        "n_trees": 0,
                        "phase1_enabled": phase1_enabled,
                        "tree_feature_mode": tree_feature_mode,
                        "tree_search_features": selected,
                        "active_retention_ratio": float(
                            retention
                        ),
                        "preparation_time": float(
                            context[
                                "preparation_time"
                            ]
                        ),
                        "fold_preparation_time": float(
                            fold[
                                "fold_preparation_time"
                            ]
                        ),
                        "phase1_time": float(
                            diagnostics.get(
                                "phase1_time",
                                0.0,
                            )
                        ),
                        "feature_pool_time": float(
                            diagnostics.get(
                                "feature_pool_time",
                                0.0,
                            )
                        ),
                        "threshold_cache_time": float(
                            diagnostics.get(
                                "threshold_cache_time",
                                0.0,
                            )
                        ),
                        "fit_wall_time": float(
                            fit_wall_time
                        ),
                        "model_total_fit_time": float(
                            diagnostics.get(
                                "total_fit_time",
                                fit_wall_time,
                            )
                        ),
                    }
                )

            except Exception as exc:
                records.append(
                    {
                        "dataset": dataset_name,
                        "task_type": "regression",
                        "experiment": (
                            "phase1_structure_snapshot"
                        ),
                        "split": int(split_id),
                        "seed": int(seed),
                        "mode": FORMAL_STRUCTURE_MODE,
                        "status": "error",
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        "candidate_pairs": int(
                            fold["candidate_pairs"]
                        ),
                        "fold_local_imputation": bool(
                            context[
                                "use_fold_local"
                            ]
                        ),
                        "final_tree_budget": 0,
                        "n_trees": 0,
                    }
                )
                raise

    return pd.DataFrame(records)


def package_version(package: str):
    try:
        return importlib.metadata.version(
            package
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def build_protocol(
    args: argparse.Namespace,
) -> dict[str, Any]:
    config_hash = sha256(
        args.regression_config
    )

    common = {
        "study": "OPNs-HybridBoost",
        "experiment": args.experiment,
        "task_type": "regression",
        "datasets": args.datasets,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "random_state": args.random_state,
        "max_folds": args.max_folds,
        "summary_std": {
            "type": (
                "sample_standard_deviation"
            ),
            "ddof": SUMMARY_DDOF,
        },
        "regression": {
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
        },
        "config": {
            "path": str(
                args.regression_config
            ),
            "sha256": config_hash,
            "expected_sha256": (
                EXPECTED_REGRESSION_CONFIG_HASH
            ),
            "matches_expected": (
                config_hash
                == EXPECTED_REGRESSION_CONFIG_HASH
            ),
        },
    }

    if args.experiment == "efficiency":
        common.update(
            {
                "protocol_name": (
                    "active_efficiency"
                ),
                "modes": list(
                    FORMAL_EFFICIENCY_MODES
                ),
                "checkpoint_step": (
                    args.checkpoint_step
                ),
                "tree_budget_cap": (
                    args.tree_budget_cap
                ),
                "staging": (
                    "single fitted model; exact "
                    "prefix predictions"
                ),
                "pairing": (
                    "full_all vs full_active "
                    "paired by dataset, split, "
                    "and n_trees from raw_curve.csv"
                ),
                "outputs": [
                    "raw_curve.csv",
                    "curve_summary.csv",
                    "active_pairwise_curve.csv",
                    "active_pairwise_summary.csv",
                ],
                "raw_curve_is_primary": True,
            }
        )
    else:
        common.update(
            {
                "protocol_name": (
                    "phase1_structure_snapshot"
                ),
                "modes": [
                    FORMAL_STRUCTURE_MODE
                ],
                "tree_budget_cap": (
                    FORMAL_STRUCTURE_TREE_BUDGET_CAP
                ),
                "phase1_enabled": True,
                "tree_feature_mode": "active",
                "interpretation": (
                    "Phase-I active-set structure "
                    "snapshot; not a boosting "
                    "learning curve"
                ),
                "outputs": [
                    "phase1_structure_snapshot.csv",
                    "phase1_structure_summary.csv",
                ],
                "raw_curve_is_primary": False,
            }
        )

    return common


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
            build_protocol(args),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": package_version(
                "numpy"
            ),
            "pandas": package_version(
                "pandas"
            ),
            "scikit-learn": package_version(
                "scikit-learn"
            ),
        },
        "arguments": {
            key: (
                str(value)
                if isinstance(
                    value,
                    Path,
                )
                else value
            )
            for key, value in vars(
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

    shutil.copyfile(
        args.regression_config,
        args.out_dir
        / args.regression_config.name,
    )


def parse_args(
    argv=None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the final "
            "OPNs-HybridBoost active-feature "
            "efficiency or Phase-I structure "
            "study."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "efficiency",
            "structure",
        ],
        default="efficiency",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
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
        default=FORMAL_CHECKPOINT_STEP,
    )

    parser.add_argument(
        "--tree-budget-cap",
        type=int,
        default=None,
        help=(
            "Optional efficiency-only cap for "
            "smoke/debug runs. Formal efficiency "
            "uses the frozen config budget."
        ),
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help=(
            "Optional per-dataset split limit for "
            "smoke/debug runs. Formal runs use "
            "all generated folds."
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
        "--regression-config",
        type=Path,
        default=DEFAULT_REGRESSION_CONFIG,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the protocol/config and "
            "write metadata without fitting."
        ),
    )

    args = parser.parse_args(
        argv
    )

    if args.datasets is None:
        if args.experiment == "efficiency":
            args.datasets = list(
                FORMAL_EFFICIENCY_DATASETS
            )
        else:
            args.datasets = list(
                FORMAL_STRUCTURE_DATASETS
            )

    if args.out_dir is None:
        suffix = (
            "reproduced_active_efficiency"
            if args.experiment
            == "efficiency"
            else "reproduced_active_structure"
        )

        args.out_dir = (
            REPO_ROOT
            / "research"
            / "hybridboost"
            / "results"
            / suffix
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
        args.tree_budget_cap
        is not None
        and args.tree_budget_cap < 0
    ):
        parser.error(
            "--tree-budget-cap cannot be negative"
        )

    if (
        args.experiment == "structure"
        and args.tree_budget_cap
        is not None
    ):
        parser.error(
            "--tree-budget-cap is efficiency-only; "
            "structure always forces tree budget 0"
        )

    if (
        args.max_folds is not None
        and args.max_folds < 1
    ):
        parser.error(
            "--max-folds must be positive"
        )

    if args.equivalence_atol < 0:
        parser.error(
            "--equivalence-atol cannot be negative"
        )

    if args.n_jobs == 0:
        parser.error(
            "--n-jobs cannot be 0"
        )

    if not args.regression_config.is_file():
        parser.error(
            "Config does not exist: "
            f"{args.regression_config}"
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
                build_protocol(args),
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "\nDry-run metadata written to: "
            f"{args.out_dir.resolve()}"
        )

        return 0

    if args.experiment == "efficiency":
        raw = run_efficiency(
            args
        )

        _write_efficiency_outputs(
            args.out_dir,
            raw,
        )

    else:
        raw = run_structure(
            args
        )

        _write_structure_outputs(
            args.out_dir,
            raw,
        )

    print(
        "\nActive-study results written to: "
        f"{args.out_dir.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
