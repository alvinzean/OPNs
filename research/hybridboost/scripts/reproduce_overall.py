from __future__ import annotations

"""Reproduce the final common-fold OPNs-HybridBoost Overall comparison.

The public entry point intentionally keeps the paper protocol compact:

* five common folds, one repeat;
* random state 161803 for all formal comparisons;
* OPNs-HybridBoost, XGBoost, LightGBM, CatBoost, and HistGB;
* the frozen paper parameter configurations in ../configs;
* Wine Quality regression uses train-fold-only median imputation;
* regression target scaling is fitted on the training fold only;
* final summaries use sample standard deviation (ddof=1).

The raw fold-level CSV files are the primary reproducibility artifacts.
"""

import argparse
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

from opns_boost.core import (
    OPNsHybridClassifier,
    OPNsHybridRegressor,
)
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

from research.hybridboost.scripts._classification_baselines import (
    align_probabilities,
    build_model as build_classic_classifier,
    classification_scores,
    merge_nested_params as merge_classification_params,
    method_available as classification_method_available,
    safe_pickle_size_mb as classification_model_size_mb,
)
from research.hybridboost.scripts._regression_baselines import (
    build_external_model as build_classic_regressor,
    map_common_budget,
    merge_nested_params as merge_regression_params,
    method_available as regression_method_available,
    regression_scores,
    safe_pickle_size_mb as regression_model_size_mb,
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
    / "regression_final.json"
)

DEFAULT_CLASSIFICATION_BASELINES = (
    CONFIG_DIR
    / "classification_baselines_final.json"
)

DEFAULT_REGRESSION_BASELINES = (
    CONFIG_DIR
    / "regression_baselines_final.json"
)


EXPECTED_CONFIG_HASHES = {
    "classification_final.json": (
        "feafdb52b8c2e3debf5c64a21d0c3f1"
        "ee3c9f6635c755dfa1d547258bc4f7976"
    ),
    "regression_final.json": (
        "6c54747483d23a8b98e90861e6c920bd"
        "977d2e7fb38c04247f78486f456b0530"
    ),
    "classification_baselines_final.json": (
        "8b930dd4dd04df44886f8886e3f0cc22"
        "2b71b0c39e37efce7eef340bc9f78890"
    ),
    "regression_baselines_final.json": (
        "f9d2368b4e2d1494e38e62fda5e371e"
        "acc3208c4f08db327feaf159e5f576e90"
    ),
}


DEFAULT_CLASSIFICATION_DATASETS = [
    "breast_cancer",
    "car",
    "iris",
    "wine",
]

DEFAULT_REGRESSION_DATASETS = [
    "airfoil",
    "boston",
    "concrete",
    "energy_cooling",
    "wine_quality",
]

DEFAULT_MODELS = [
    "opns",
    "xgboost",
    "lightgbm",
    "catboost",
    "histgb",
]

EXTERNAL_MODELS = {
    "xgboost",
    "lightgbm",
    "catboost",
    "histgb",
}

ALL_MODELS = {
    "opns",
    *EXTERNAL_MODELS,
}

METHOD_LABELS = {
    "opns": "OPNs-HybridBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "histgb": "HistGB",
}

FORMAL_RANDOM_STATE = 161803

FORMAL_N_SPLITS = 5

FORMAL_N_REPEATS = 1

SUMMARY_DDOF = 1


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
            f"Expected a JSON object: {path}"
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


def _check_models(
    models: list[str],
    *,
    skip_missing: bool,
) -> list[str]:
    normalized = [
        value.lower()
        for value
        in models
    ]

    unknown = sorted(
        set(normalized)
        - ALL_MODELS
    )

    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. "
            f"Available: {sorted(ALL_MODELS)}"
        )

    available = []

    missing = []

    for method in normalized:
        if method == "opns":
            available.append(
                method
            )
            continue

        ok, package = (
            classification_method_available(
                method
            )
        )

        if ok:
            available.append(
                method
            )

        else:
            missing.append(
                f"{method} ({package})"
            )

    if missing and not skip_missing:
        raise RuntimeError(
            "Missing optional baseline packages: "
            + ", ".join(
                missing
            )
            + ". Install the HybridBoost "
              "benchmark requirements or pass "
              "--skip-missing-baselines."
        )

    return available


def _summary(
    raw: pd.DataFrame,
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
        method,
    ), group in successful.groupby(
        [
            "dataset",
            "method",
        ],
        sort=False,
    ):
        row = {
            "dataset": dataset,
            "method": method,
            "folds": int(
                len(
                    group
                )
            ),
        }

        for metric in metrics:
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


def _classification_record(
    *,
    dataset_name: str,
    method_key: str,
    split_id: int,
    seed: int,
    y_test,
    y_pred,
    y_proba,
    classes,
    fit_time: float,
    predict_time: float,
    model,
    params,
):
    scores = classification_scores(
        np.asarray(
            y_test,
            dtype=int,
        ),
        np.asarray(
            y_pred,
            dtype=int,
        ),
        np.asarray(
            y_proba,
            dtype=float,
        ),
        np.asarray(
            classes
        ),
    )

    n_test = max(
        1,
        len(
            y_test
        ),
    )

    return {
        "dataset": dataset_name,
        "task_type": (
            "classification"
        ),
        "method": (
            METHOD_LABELS[
                method_key
            ]
        ),
        "split": split_id,
        "seed": seed,
        "status": "ok",
        "error": "",
        **scores,
        "fit_time": float(
            fit_time
        ),
        "predict_time": float(
            predict_time
        ),
        "predict_us_per_sample": (
            float(
                predict_time
                / n_test
                * 1.0e6
            )
        ),
        "model_size_mb": (
            classification_model_size_mb(
                model
            )
        ),
        "params_json": (
            json.dumps(
                params,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        ),
    }


def run_classification(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    opns_config = load_json(
        args.classification_config
    )

    baseline_config = load_json(
        args.classification_baseline_config
    )

    models = _check_models(
        args.models,
        skip_missing=(
            args.skip_missing_baselines
        ),
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

        X_frame = (
            dataset.X
            .reset_index(
                drop=True
            )
        )

        X_array = (
            X_frame.to_numpy(
                dtype=float
            )
        )

        y = np.asarray(
            dataset.y,
            dtype=int,
        ).reshape(-1)

        classes = np.unique(
            y
        )

        X_opns = None

        if "opns" in models:
            X_opns, _ = (
                make_opns_features(
                    X_frame,
                    mode="combinations",
                    scale_features="none",
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

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]

            for method in models:
                if method == "opns":
                    params = (
                        resolve_dataset_params(
                            opns_config,
                            dataset_name,
                        )
                    )

                    params = dict(
                        params
                    )

                    params.update(
                        random_state=seed,
                        verbose=False,
                    )

                    model = (
                        OPNsHybridClassifier(
                            **params
                        )
                    )

                    X_train = X_opns[
                        train_idx
                    ]

                    X_test = X_opns[
                        test_idx
                    ]

                    with _thread_context(
                        args.n_jobs
                    ):
                        start = (
                            time.perf_counter()
                        )

                        model.fit(
                            X_train,
                            y_train,
                            X_val=X_test,
                            y_val=y_test,
                        )

                        fit_time = (
                            time.perf_counter()
                            - start
                        )

                        start = (
                            time.perf_counter()
                        )

                        y_pred = np.asarray(
                            model.predict(
                                X_test
                            )
                        ).reshape(-1)

                        y_proba = (
                            align_probabilities(
                                model,
                                model.predict_proba(
                                    X_test
                                ),
                                classes,
                            )
                        )

                        predict_time = (
                            time.perf_counter()
                            - start
                        )

                else:
                    overrides = (
                        merge_classification_params(
                            baseline_config,
                            dataset_name,
                            method,
                        )
                    )

                    model = (
                        build_classic_classifier(
                            method,
                            n_classes=len(
                                classes
                            ),
                            n_estimators=60,
                            learning_rate=0.1,
                            max_depth=3,
                            l2_leaf_reg=1.0,
                            random_state=seed,
                            n_jobs=args.n_jobs,
                            overrides=overrides,
                        )
                    )

                    if (
                        method
                        == "lightgbm"
                    ):
                        X_train = (
                            X_frame.iloc[
                                train_idx
                            ].copy()
                        )

                        X_test = (
                            X_frame.iloc[
                                test_idx
                            ].copy()
                        )

                    else:
                        X_train = X_array[
                            train_idx
                        ]

                        X_test = X_array[
                            test_idx
                        ]

                    with _thread_context(
                        args.n_jobs
                    ):
                        start = (
                            time.perf_counter()
                        )

                        model.fit(
                            X_train,
                            y_train,
                        )

                        fit_time = (
                            time.perf_counter()
                            - start
                        )

                        start = (
                            time.perf_counter()
                        )

                        y_pred = np.asarray(
                            model.predict(
                                X_test
                            )
                        ).reshape(-1)

                        y_proba = (
                            align_probabilities(
                                model,
                                model.predict_proba(
                                    X_test
                                ),
                                classes,
                            )
                        )

                        predict_time = (
                            time.perf_counter()
                            - start
                        )

                    params = overrides

                records.append(
                    _classification_record(
                        dataset_name=(
                            dataset_name
                        ),
                        method_key=method,
                        split_id=split_id,
                        seed=seed,
                        y_test=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        classes=classes,
                        fit_time=fit_time,
                        predict_time=(
                            predict_time
                        ),
                        model=model,
                        params=params,
                    )
                )

    raw = pd.DataFrame(
        records
    )

    summary = _summary(
        raw,
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "LogLoss",
            "AUC",
            "fit_time",
            "predict_time",
            "predict_us_per_sample",
            "model_size_mb",
        ],
    )

    return (
        raw,
        summary,
    )


def run_regression(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    opns_config = load_json(
        args.regression_config
    )

    baseline_config = load_json(
        args.regression_baseline_config
    )

    models = _check_models(
        args.models,
        skip_missing=(
            args.skip_missing_baselines
        ),
    )

    records = []

    for dataset_name in (
        args.regression_datasets
    ):
        fold_local = (
            uses_fold_local_imputation(
                dataset_name
            )
        )

        dataset = (
            load_regression_dataset(
                dataset_name,
                data_root=args.data_root,
                defer_imputation=(
                    fold_local
                ),
            )
        )

        y_all = np.asarray(
            dataset.y,
            dtype=float,
        ).reshape(-1)

        global_range = float(
            np.max(
                y_all
            )
            - np.min(
                y_all
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

        opns_params = (
            resolve_dataset_params(
                opns_config,
                dataset_name,
            )
        )

        common_budget = (
            map_common_budget(
                opns_params
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

            prep_start = (
                time.perf_counter()
            )

            prepared = (
                prepare_regression_fold(
                    dataset,
                    train_idx,
                    test_idx,
                    need_opns=(
                        "opns"
                        in models
                    ),
                    fold_local_imputation=(
                        fold_local
                    ),
                    pairing="combinations",
                    scale_features="none",
                )
            )

            prep_time = (
                time.perf_counter()
                - prep_start
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

            candidate_pairs = (
                len(
                    prepared.pair_features
                )
                // 2
            )

            for method in models:
                if method == "opns":
                    params = dict(
                        opns_params
                    )

                    params.update(
                        random_state=seed,
                        verbose=False,
                        early_stopping_rounds=None,
                    )

                    backend = str(
                        params.get(
                            "prediction_backend",
                            "spectral_batch",
                        )
                    )

                    model = (
                        OPNsHybridRegressor(
                            **params
                        )
                    )

                    y_model = (
                        to_opns_target(
                            y_train_scaled
                        )
                    )

                    with _thread_context(
                        args.n_jobs
                    ):
                        start = (
                            time.perf_counter()
                        )

                        model.fit(
                            prepared.X_train_opns,
                            y_model,
                        )

                        fit_time = (
                            time.perf_counter()
                            - start
                        )

                        start = (
                            time.perf_counter()
                        )

                        pred_scaled = (
                            np.asarray(
                                model.predict(
                                    prepared.X_test_opns,
                                    item=1,
                                    backend=backend,
                                ),
                                dtype=float,
                            ).reshape(-1)
                        )

                        predict_time = (
                            time.perf_counter()
                            - start
                        )

                    model_size = (
                        regression_model_size_mb(
                            model
                        )
                    )

                else:
                    overrides = (
                        merge_regression_params(
                            baseline_config,
                            dataset_name,
                            method,
                        )
                    )

                    model = (
                        build_classic_regressor(
                            method,
                            common_budget,
                            seed,
                            args.n_jobs,
                            overrides,
                        )
                    )

                    if (
                        method
                        == "lightgbm"
                    ):
                        X_train = (
                            prepared.X_train_frame.copy()
                        )

                        X_test = (
                            prepared.X_test_frame.copy()
                        )

                    else:
                        X_train = (
                            prepared.X_train_frame
                            .to_numpy(
                                dtype=float
                            )
                        )

                        X_test = (
                            prepared.X_test_frame
                            .to_numpy(
                                dtype=float
                            )
                        )

                    with _thread_context(
                        args.n_jobs
                    ):
                        start = (
                            time.perf_counter()
                        )

                        model.fit(
                            X_train,
                            y_train_scaled,
                        )

                        fit_time = (
                            time.perf_counter()
                            - start
                        )

                        start = (
                            time.perf_counter()
                        )

                        pred_scaled = (
                            np.asarray(
                                model.predict(
                                    X_test
                                ),
                                dtype=float,
                            ).reshape(-1)
                        )

                        predict_time = (
                            time.perf_counter()
                            - start
                        )

                    params = overrides

                    model_size = (
                        regression_model_size_mb(
                            model
                        )
                    )

                y_pred_raw = (
                    target_scaler
                    .inverse_transform(
                        pred_scaled.reshape(
                            -1,
                            1,
                        )
                    )
                    .reshape(-1)
                )

                scores = regression_scores(
                    y_test_raw,
                    y_pred_raw,
                    global_range,
                )

                records.append(
                    {
                        "dataset": (
                            dataset_name
                        ),
                        "task_type": (
                            "regression"
                        ),
                        "method": (
                            METHOD_LABELS[
                                method
                            ]
                        ),
                        "split": split_id,
                        "seed": seed,
                        "status": "ok",
                        "error": "",
                        "n_samples": len(
                            y_all
                        ),
                        "n_train": len(
                            train_idx
                        ),
                        "n_test": len(
                            test_idx
                        ),
                        "n_original_features": (
                            prepared.X_train_frame
                            .shape[
                                1
                            ]
                        ),
                        "candidate_pairs": (
                            candidate_pairs
                            if method
                            == "opns"
                            else 0
                        ),
                        "fold_local_imputation": (
                            fold_local
                        ),
                        "preprocess_time": (
                            float(
                                prep_time
                            )
                        ),
                        **scores,
                        "fit_time": (
                            float(
                                fit_time
                            )
                        ),
                        "predict_time": (
                            float(
                                predict_time
                            )
                        ),
                        "predict_us_per_sample": (
                            float(
                                predict_time
                                / max(
                                    1,
                                    len(
                                        test_idx
                                    ),
                                )
                                * 1.0e6
                            )
                        ),
                        "model_size_mb": (
                            model_size
                        ),
                        "params_json": (
                            json.dumps(
                                params,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            )
                        ),
                    }
                )

    raw = pd.DataFrame(
        records
    )

    summary = _summary(
        raw,
        [
            "RMSE",
            "NRMSE_range",
            "MAE",
            "R2",
            "preprocess_time",
            "fit_time",
            "predict_time",
            "predict_us_per_sample",
            "model_size_mb",
        ],
    )

    return (
        raw,
        summary,
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
):
    config_paths = {
        "classification": (
            args.classification_config
        ),
        "regression": (
            args.regression_config
        ),
        "classification_baselines": (
            args.classification_baseline_config
        ),
        "regression_baselines": (
            args.regression_baseline_config
        ),
    }

    return {
        "study": (
            "OPNs-HybridBoost"
        ),
        "experiment": (
            "overall_common_fold"
        ),
        "task": args.task,
        "classification_datasets": (
            args.classification_datasets
        ),
        "regression_datasets": (
            args.regression_datasets
        ),
        "models": args.models,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "random_state": (
            args.random_state
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
            "split": (
                "RepeatedStratifiedKFold"
            ),
            "feature_scaling": (
                "none"
            ),
        },
        "regression": {
            "split": (
                "RepeatedKFold"
            ),
            "target_scaling": (
                "MinMaxScaler fitted on "
                "training targets only"
            ),
            "wine_quality_imputation": (
                "training-fold median only"
            ),
        },
        "configs": {
            key: {
                "path": str(
                    path
                ),
                "sha256": sha256(
                    path
                ),
            }
            for key, path
            in config_paths.items()
        },
        "raw_fold_rows_are_primary": (
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

    protocol = build_protocol(
        args
    )

    (
        args.out_dir
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
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
            "xgboost": (
                package_version(
                    "xgboost"
                )
            ),
            "lightgbm": (
                package_version(
                    "lightgbm"
                )
            ),
            "catboost": (
                package_version(
                    "catboost"
                )
            ),
        },
        "arguments": {
            key: str(
                value
            )
            if isinstance(
                value,
                Path,
            )
            else value
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
        args.classification_baseline_config,
        args.regression_baseline_config,
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
            "Reproduce the final OPNs-HybridBoost "
            "common-fold Overall comparison."
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
        "--models",
        nargs="+",
        default=list(
            DEFAULT_MODELS
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
        "--classification-baseline-config",
        type=Path,
        default=(
            DEFAULT_CLASSIFICATION_BASELINES
        ),
    )

    parser.add_argument(
        "--regression-baseline-config",
        type=Path,
        default=(
            DEFAULT_REGRESSION_BASELINES
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
            / "reproduced_overall"
        ),
    )

    parser.add_argument(
        "--skip-missing-baselines",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the frozen protocol/configs "
            "and write metadata without fitting models."
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

    if args.n_jobs == 0:
        parser.error(
            "--n-jobs cannot be 0"
        )

    args.models = [
        value.lower()
        for value
        in args.models
    ]

    _check_models(
        args.models,
        skip_missing=(
            args.skip_missing_baselines
            or args.dry_run
        ),
    )

    for path in (
        args.classification_config,
        args.regression_config,
        args.classification_baseline_config,
        args.regression_baseline_config,
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
            f"\nDry-run metadata written to: "
            f"{args.out_dir.resolve()}"
        )

        return 0

    raw_frames = []

    summary_frames = []

    if args.task in {
        "all",
        "classification",
    }:
        raw, summary = (
            run_classification(
                args
            )
        )

        raw.to_csv(
            args.out_dir
            / "classification_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            args.out_dir
            / "classification_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raw_frames.append(
            raw
        )

        summary_frames.append(
            summary.assign(
                task_type=(
                    "classification"
                )
            )
        )

    if args.task in {
        "all",
        "regression",
    }:
        raw, summary = (
            run_regression(
                args
            )
        )

        raw.to_csv(
            args.out_dir
            / "regression_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            args.out_dir
            / "regression_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raw_frames.append(
            raw
        )

        summary_frames.append(
            summary.assign(
                task_type=(
                    "regression"
                )
            )
        )

    if raw_frames:
        pd.concat(
            raw_frames,
            ignore_index=True,
            sort=False,
        ).to_csv(
            args.out_dir
            / "overall_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if summary_frames:
        pd.concat(
            summary_frames,
            ignore_index=True,
            sort=False,
        ).to_csv(
            args.out_dir
            / "overall_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        f"\nResults written to: "
        f"{args.out_dir.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
