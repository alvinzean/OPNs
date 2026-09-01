from __future__ import annotations

"""Frozen regression warm-start helper semantics migrated from the verified paper runner."""

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from opns_pack.opns import OPNs
import opns_pack.opns_np as op
from opns_boost.core import OPNsHybridRegressor, extract_real
from opns_boost.data import load_regression_dataset, make_opns_features, prepare_regression_fold
from opns_boost.experiment import resolve_dataset_params
from research.hybridboost.ablations.tree_only import OPNsTreeOnlyRegressor


def build_model(mode: str, params: dict[str, Any], seed: int):
    model_params = dict(params)
    model_params.pop("warm_start_mode", None)
    model_params.update(
        random_state=int(seed),
        early_stopping_rounds=None,
        verbose=False,
        prediction_backend="object_batch",
    )

    if mode == "full_all":
        model_params["tree_feature_mode"] = "all"
        return OPNsHybridRegressor(**model_params)
    if mode == "full_active":
        model_params["tree_feature_mode"] = "active"
        return OPNsHybridRegressor(**model_params)
    if mode == "tree_only_all":
        model_params["tree_feature_mode"] = "all"
        return OPNsTreeOnlyRegressor(warm_start_mode="constant", **model_params)
    raise KeyError(f"Unknown mode: {mode}")


def staged_predictions(model, X, checkpoints: set[int]) -> dict[int, np.ndarray]:
    """Evaluate all requested prefixes in one exact object-batch pass."""
    checkpoints = {int(value) for value in checkpoints}
    n_trees = len(model.trees_)
    invalid = sorted(value for value in checkpoints if value < 0 or value > n_trees)
    if invalid:
        raise ValueError(f"Invalid tree checkpoints {invalid}; fitted trees={n_trees}.")

    if getattr(model, "base_model_", None) is None:
        F = model._repeat_opns(model.base_constant_, X.shape[0])
    else:
        X_poly = model._expand_features(X)
        X_poly_scaled = model.base_scaler_.transform(X_poly)
        F = op.dot(X_poly_scaled, model.base_model_.coef_) + model.base_model_.intercept_

    predictions: dict[int, np.ndarray] = {}
    if 0 in checkpoints:
        predictions[0] = extract_real(F, item=1).copy()

    for completed, (tree, lr_opns) in enumerate(model.trees_, start=1):
        if hasattr(model, "_tree_prediction"):
            tree_prediction = model._tree_prediction(tree, X, "object_batch")
        else:
            tree_prediction = tree.predict_object_batch(X)
        F = F + tree_prediction * lr_opns
        if completed in checkpoints:
            predictions[completed] = extract_real(F, item=1).copy()

    missing = checkpoints.difference(predictions)
    if missing:
        raise RuntimeError(f"Failed to produce staged predictions for {sorted(missing)}.")
    return predictions


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def checkpoint_grid(n_trees: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    values = set(range(0, n_trees + 1, step))
    values.update({0, n_trees, min(10, n_trees), min(30, n_trees), min(60, n_trees), min(120, n_trees)})
    values.add(int(round(n_trees / 2)))
    return sorted(value for value in values if 0 <= value <= n_trees)


def time_to_checkpoint(model, n_trees: int) -> float:
    tree_times = np.asarray(getattr(model, "tree_iteration_times_", []), dtype=float)
    if n_trees == 0:
        if getattr(model, "base_model_", None) is None:
            return float(getattr(model, "cold_start_time_", 0.0))
        return float(getattr(model, "phase1_time_", 0.0))

    initialization = (
        float(getattr(model, "cold_start_time_", 0.0))
        if getattr(model, "base_model_", None) is None
        else float(getattr(model, "phase1_time_", 0.0))
    )
    setup = float(getattr(model, "feature_pool_time_", 0.0)) + float(
        getattr(model, "threshold_cache_time_", 0.0)
    )
    return initialization + setup + float(np.sum(tree_times[:n_trees]))


def cumulative_count(model, attribute: str, n_trees: int) -> int:
    values = getattr(model, attribute, [])
    return int(sum(int(value) for value in values[:n_trees]))
