"""Small experiment runners for OPNs-HybridBoost.

These functions are deliberately lightweight.  They provide a reproducible
protocol without hiding model logic in notebooks or per-dataset test scripts.
"""
from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd

from .core import OPNsHybridClassifier, OPNsHybridRegressor, extract_real
from .data import prepare_classification_data, prepare_regression_data, repeated_splits
from .metrics import classification_metrics, regression_metrics, summarize_records


LOG_PARAM_NAMES = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "l2_leaf_reg",
    "feature_filter_ratio",
    "poly_degree",
    "logistic_lr",
    "logistic_max_iter",
    "use_trig",
    "lasso_alpha",
    "lasso_max_iter",
    "active_threshold",
    "random_strength",
    "lr_decay_type",
    "early_stopping_rounds",
    "max_thresholds",
    "tree_feature_mode",
    "active_plus_ratio",
    "residual_gain_ratio",
    "residual_gain_max_thresholds",
    "residual_gain_gate_quantile",
    "residual_gain_stability_repeats",
    "residual_gain_subsample",
    "residual_gain_min_frequency",
    "residual_gain_crossfit_repeats",
    "residual_gain_crossfit_validation_fraction",
    "residual_gain_crossfit_min_frequency",
    "progress_interval",
    "split_score_backend",
    "threshold_backend",
    "threshold_sampling",
    "prediction_backend",
]


def resolve_dataset_params(model_params: dict | None, dataset_name: str) -> dict:
    """Resolve model parameters for one dataset.

    Supported formats:
    1. Flat params:
       {"n_estimators": 200, "learning_rate": 0.1}

    2. Per-dataset params:
       {
         "__default__": {...},
         "concrete": {...},
         "airfoil": {...}
       }
    """
    if not model_params:
        return {}

    has_nested = any(isinstance(v, dict) for v in model_params.values())

    if not has_nested:
        return dict(model_params)

    params = dict(model_params.get("__default__", {}))
    params.update(model_params.get(dataset_name, {}))
    return params


def serialize_param_value(v):
    """Convert parameter values to CSV-friendly values for logging only."""
    if v is None:
        return "None"
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def run_opns_regression_cv(
    datasets: list[str],
    model_params: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    pairing: Literal["combinations", "permutations"] = "combinations",
    verbose: bool = False,
    progress_interval: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_params = model_params or {}
    if progress_interval <= 0:
        raise ValueError("progress_interval must be a positive integer.")

    records = []
    total_folds = n_splits * n_repeats
    total_datasets = len(datasets)
    for dataset_pos, dataset_name in enumerate(datasets, start=1):
        dataset_started = time.perf_counter()
        if verbose:
            print(
                f"\n[dataset {dataset_pos}/{total_datasets}] {dataset_name}: "
                "preparing OPNs pair features...",
                flush=True,
            )
        prepared = prepare_regression_data(dataset_name, pairing=pairing)
        if verbose:
            print(
                f"[dataset {dataset_pos}/{total_datasets}] {dataset_name}: "
                f"prepared {prepared.X_opns.shape[0]} samples, "
                f"{prepared.candidate_pairs} candidate pairs in "
                f"{time.perf_counter() - dataset_started:.2f}s.",
                flush=True,
            )
        y_true_scaled = extract_real(prepared.y_model, item=1)
        for split_id, (train_idx, test_idx) in enumerate(repeated_splits(y_true_scaled, "regression", n_splits, n_repeats, random_state)):
            X_train, X_test = prepared.X_opns[train_idx], prepared.X_opns[test_idx]
            y_train, y_test = prepared.y_model[train_idx], prepared.y_model[test_idx]
            # model = OPNsHybridRegressor(**{**model_params, "random_state": random_state + split_id, "verbose": False})
            dataset_params = resolve_dataset_params(model_params, dataset_name)
            if verbose:
                print(
                    f"  [fold {split_id + 1}/{total_folds}] "
                    f"train={len(train_idx)}, test={len(test_idx)}, "
                    f"seed={random_state + split_id}",
                    flush=True,
                )
            model = OPNsHybridRegressor(
                **{
                    **dataset_params,
                    "random_state": random_state + split_id,
                    "verbose": verbose,
                    "progress_interval": progress_interval,
                }
            )
            t0 = time.perf_counter()
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            train_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = model.predict(X_test, item=1)
            infer_time = time.perf_counter() - t1
            metrics = regression_metrics(extract_real(y_test, item=1), pred)
            diag = model.get_diagnostics()
            # records.append({
            #     "dataset": dataset_name,
            #     "task_type": "regression",
            #     "method": "OPNs-HybridBoost",
            #     "split": split_id,
            #     "seed": random_state + split_id,
            #     "candidate_pairs": prepared.candidate_pairs,
            #     "train_time": train_time,
            #     "inference_time": infer_time,
            #     **metrics,
            #     **{k: v for k, v in diag.items() if k != "active_features"},
            # })
            record = {
                "dataset": dataset_name,
                "task_type": "regression",
                "method": "OPNs-HybridBoost",
                "split": split_id,
                "seed": random_state + split_id,
                "candidate_pairs": prepared.candidate_pairs,
                "train_time": train_time,
                "inference_time": infer_time,
                **metrics,
                **{k: v for k, v in diag.items() if k != "active_features"},
            }

            # record.update({
            #     f"param_{k}": serialize_param_value(v)
            #     for k, v in dataset_params.items()
            #     if isinstance(v, (int, float, str, bool)) or v is None
            # })
            record.update({
                f"param_{name}": serialize_param_value(getattr(model, name, None))
                for name in LOG_PARAM_NAMES
            })

            records.append(record)
            if verbose:
                print(
                    f"  [fold {split_id + 1}/{total_folds}] completed: "
                    f"RMSE={metrics['RMSE']:.6f}, "
                    f"R2={metrics['R2']:.6f}, "
                    f"train={train_time:.2f}s, infer={infer_time:.4f}s, "
                    f"search_features={diag.get('tree_search_features')}.",
                    flush=True,
                )

    raw = pd.DataFrame(records)
    return raw, summarize_records(records)


def run_opns_classification_cv(
    datasets: list[str],
    model_params: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    pairing: Literal["combinations", "permutations"] = "combinations",
    max_original_features: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_params = model_params or {}
    records = []
    for dataset_name in datasets:
        prepared = prepare_classification_data(dataset_name, pairing=pairing, max_original_features=max_original_features)
        y = np.asarray(prepared.y_model)
        for split_id, (train_idx, test_idx) in enumerate(repeated_splits(y, "classification", n_splits, n_repeats, random_state)):
            X_train, X_test = prepared.X_opns[train_idx], prepared.X_opns[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            dataset_params = resolve_dataset_params(model_params, dataset_name)
            model = OPNsHybridClassifier(
                **{
                    **dataset_params,
                    "random_state": random_state + split_id,
                    "verbose": False
                }
            )
            t0 = time.perf_counter()
            # model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
            infer_time = time.perf_counter() - t1
            metrics = classification_metrics(y_test, pred, proba)
            diag = model.get_diagnostics()
            # records.append({
            #     "dataset": dataset_name,
            #     "task_type": "classification",
            #     "method": "OPNs-HybridBoost",
            #     "split": split_id,
            #     "seed": random_state + split_id,
            #     "candidate_pairs": prepared.candidate_pairs,
            #     "train_time": train_time,
            #     "inference_time": infer_time,
            #     **metrics,
            #     **{k: v for k, v in diag.items() if k != "per_class"},
            # })
            record = {
                "dataset": dataset_name,
                "task_type": "classification",
                "method": "OPNs-HybridBoost",
                "split": split_id,
                "seed": random_state + split_id,
                "candidate_pairs": prepared.candidate_pairs,
                "train_time": train_time,
                "inference_time": infer_time,
                **metrics,
                **{k: v for k, v in diag.items() if k != "active_features"},
            }

            # record.update({
            #     f"param_{k}": serialize_param_value(v)
            #     for k, v in dataset_params.items()
            #     if isinstance(v, (int, float, str, bool)) or v is None
            # })
            record.update({
                f"param_{name}": serialize_param_value(getattr(model, name, None))
                for name in LOG_PARAM_NAMES
            })

            records.append(record)

    raw = pd.DataFrame(records)
    return raw, summarize_records(records)
