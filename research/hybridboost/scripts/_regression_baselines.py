from __future__ import annotations

"""Frozen classic-regression helpers used by reproduce_overall.py."""

import importlib.metadata
import math
import pickle

from typing import Any

import numpy as np

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def merge_nested_params(config: dict, dataset: str, method: str) -> dict:
    """Resolve optional baseline overrides.

    Supported structure::

        {
          "__default__": {
            "xgboost": {"n_estimators": 300}
          },
          "concrete": {
            "xgboost": {"max_depth": 5}
          }
        }
    """
    result: dict[str, Any] = {}
    default_block = config.get("__default__", {})
    if isinstance(default_block, dict):
        default_method = default_block.get(method, {})
        if isinstance(default_method, dict):
            result.update(default_method)
    dataset_block = config.get(dataset, {})
    if isinstance(dataset_block, dict):
        dataset_method = dataset_block.get(method, {})
        if isinstance(dataset_method, dict):
            result.update(dataset_method)
    return result


def method_available(method: str) -> tuple[bool, str | None]:
    package_map = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }
    package = package_map.get(method)
    if package is None:
        return True, None
    version = package_version(package)
    if version is None:
        return False, package
    return True, None


def map_common_budget(opns_params: dict) -> dict[str, Any]:
    """Map the OPNs budget to parameters shared by other boosting models."""
    depth = int(opns_params.get("max_depth", 5))
    return {
        "n_estimators": int(opns_params.get("n_estimators", 300)),
        "learning_rate": float(opns_params.get("learning_rate", 0.1)),
        "max_depth": depth,
        "l2_leaf_reg": float(opns_params.get("l2_leaf_reg", 1.0)),
        "num_leaves": min(2 ** max(depth, 1), 255),
    }


def build_external_model(
    method: str,
    common: dict[str, Any],
    random_state: int,
    n_jobs: int,
    overrides: dict,
):
    n_estimators = common["n_estimators"]
    learning_rate = common["learning_rate"]
    max_depth = common["max_depth"]
    l2 = common["l2_leaf_reg"]

    if method == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc
        params = {
            "objective": "reg:squarederror",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "reg_lambda": l2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": 0,
        }
        params.update(overrides)
        return XGBRegressor(**params)

    if method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed") from exc
        params = {
            "objective": "regression",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": common["num_leaves"],
            "reg_lambda": l2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": -1,
        }
        params.update(overrides)
        return LGBMRegressor(**params)

    if method == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("catboost is not installed") from exc
        params = {
            "iterations": n_estimators,
            "learning_rate": learning_rate,
            "depth": max_depth,
            "l2_leaf_reg": l2,
            "loss_function": "RMSE",
            "random_seed": random_state,
            "thread_count": n_jobs,
            "verbose": False,
            "allow_writing_files": False,
        }
        params.update(overrides)
        return CatBoostRegressor(**params)

    if method == "histgb":
        params = {
            "max_iter": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "l2_regularization": l2,
            "early_stopping": False,
            "random_state": random_state,
        }
        params.update(overrides)
        return HistGradientBoostingRegressor(**params)

    if method == "sklearn_gbr":
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
            "loss": "squared_error",
        }
        params.update(overrides)
        return GradientBoostingRegressor(**params)

    raise KeyError(f"Unknown external method: {method}")


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray, global_range: float) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "RMSE": rmse,
        "NRMSE_range": rmse / global_range if global_range > 0 else math.nan,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_pickle_size_mb(model: Any) -> float:
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / (1024.0 * 1024.0)
    except Exception:
        return math.nan
