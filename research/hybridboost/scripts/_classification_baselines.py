from __future__ import annotations

"""Frozen classic-classification helpers used by reproduce_overall.py."""

import importlib.metadata
import math
import pickle

from typing import Any

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def merge_nested_params(
    config: dict[str, Any],
    dataset: str,
    method: str,
) -> dict[str, Any]:
    """Merge optional baseline overrides.

    Supported structure:
    {
      "__default__": {
        "__default__": {...},
        "xgboost": {...}
      },
      "iris": {
        "__default__": {...},
        "xgboost": {...}
      }
    }
    """
    merged: dict[str, Any] = {}

    default_block = config.get("__default__", {})
    if isinstance(default_block, dict):
        common = default_block.get("__default__", {})
        if isinstance(common, dict):
            merged.update(common)
        method_block = default_block.get(method, {})
        if isinstance(method_block, dict):
            merged.update(method_block)

    dataset_block = config.get(dataset, {})
    if isinstance(dataset_block, dict):
        common = dataset_block.get("__default__", {})
        if isinstance(common, dict):
            merged.update(common)
        method_block = dataset_block.get(method, {})
        if isinstance(method_block, dict):
            merged.update(method_block)

    return merged


def method_available(method: str) -> tuple[bool, str | None]:
    package_map = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }
    package = package_map.get(method)
    if package is None:
        return True, None
    return package_version(package) is not None, package


def build_model(
    method: str,
    *,
    n_classes: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    l2_leaf_reg: float,
    random_state: int,
    n_jobs: int,
    overrides: dict[str, Any],
):
    multiclass = n_classes > 2
    max_leaf_nodes = max(2, min(2 ** max_depth, 255))

    if method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc

        params: dict[str, Any] = {
            "objective": "multi:softprob" if multiclass else "binary:logistic",
            "eval_metric": "mlogloss" if multiclass else "logloss",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "reg_lambda": l2_leaf_reg,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": 0,
        }
        if multiclass:
            params["num_class"] = n_classes
        params.update(overrides)
        return XGBClassifier(**params)

    if method == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed") from exc

        params = {
            "objective": "multiclass" if multiclass else "binary",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": max_leaf_nodes,
            "reg_lambda": l2_leaf_reg,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": -1,
        }
        if multiclass:
            params["num_class"] = n_classes
        params.update(overrides)
        return LGBMClassifier(**params)

    if method == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is not installed") from exc

        params = {
            "iterations": n_estimators,
            "learning_rate": learning_rate,
            "depth": max_depth,
            "l2_leaf_reg": l2_leaf_reg,
            "loss_function": "MultiClass" if multiclass else "Logloss",
            "random_seed": random_state,
            "thread_count": n_jobs,
            "verbose": False,
            "allow_writing_files": False,
        }
        params.update(overrides)
        return CatBoostClassifier(**params)

    if method == "histgb":
        params = {
            "max_iter": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_leaf_reg,
            "early_stopping": False,
            "random_state": random_state,
        }
        params.update(overrides)
        return HistGradientBoostingClassifier(**params)

    raise KeyError(f"Unknown classification baseline: {method}")


def align_probabilities(
    model: Any,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])

    model_classes = np.asarray(getattr(model, "classes_", classes))
    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    class_to_column = {value: index for index, value in enumerate(classes.tolist())}

    for source_column, class_value in enumerate(model_classes.tolist()):
        if class_value not in class_to_column:
            raise RuntimeError(
                f"Model returned an unknown class {class_value!r}; expected {classes.tolist()}."
            )
        aligned[:, class_to_column[class_value]] = probabilities[:, source_column]

    eps = np.finfo(float).eps
    aligned = np.clip(aligned, eps, 1.0)
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Predicted probability rows must have positive mass.")
    return aligned / row_sums


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    binary = len(classes) == 2
    average = "binary" if binary else "macro"

    scores = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "LogLoss": float(log_loss(y_true, y_proba, labels=classes)),
    }

    try:
        if binary:
            scores["AUC"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            scores["AUC"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        scores["AUC"] = math.nan

    return scores


def safe_pickle_size_mb(model: Any) -> float:
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / (
            1024.0 * 1024.0
        )
    except Exception:
        return math.nan
