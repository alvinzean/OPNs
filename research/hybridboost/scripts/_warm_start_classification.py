from __future__ import annotations

"""Frozen classification warm-start helper semantics migrated from the verified paper runner."""

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
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import opns_pack.opns_np as op
from opns_boost.core import OPNsHybridClassifier, OPNsLogLoss
from opns_boost.data import load_classification_dataset, make_opns_features
from opns_boost.experiment import resolve_dataset_params


def build_model(mode: str, params: dict[str, Any], seed: int) -> OPNsHybridClassifier:
    model_params = dict(params)
    model_params.update(
        random_state=int(seed),
        early_stopping_rounds=None,
        verbose=False,
    )
    if mode == "full_all":
        model_params.update(phase1_enabled=True, tree_feature_mode="all")
    elif mode == "tree_only_all":
        model_params.update(phase1_enabled=False, tree_feature_mode="all")
    elif mode == "full_active":
        model_params.update(phase1_enabled=True, tree_feature_mode="active")
    else:
        raise KeyError(f"Unknown mode: {mode}")
    return OPNsHybridClassifier(**model_params)


def checkpoint_grid(n_rounds: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    values = set(range(0, n_rounds + 1, step))
    values.update({0, n_rounds, min(10, n_rounds), min(30, n_rounds), min(60, n_rounds)})
    values.add(int(round(n_rounds / 2)))
    return sorted(v for v in values if 0 <= v <= n_rounds)


def _binary_staged_logits(estimator, X, checkpoints: set[int]) -> dict[int, np.ndarray]:
    checkpoints = {int(v) for v in checkpoints}
    n_trees = len(estimator.trees_)
    invalid = sorted(v for v in checkpoints if v < 0 or v > n_trees)
    if invalid:
        raise ValueError(f"Invalid checkpoints {invalid}; fitted trees={n_trees}.")

    F = estimator._initial_logits(X)
    output: dict[int, np.ndarray] = {}
    if 0 in checkpoints:
        output[0] = OPNsLogLoss.logits_to_real(F).copy()
    for completed, (tree, lr_opns) in enumerate(estimator.trees_, start=1):
        F = F + tree.predict_object_batch(X) * lr_opns
        if completed in checkpoints:
            output[completed] = OPNsLogLoss.logits_to_real(F).copy()
    missing = checkpoints.difference(output)
    if missing:
        raise RuntimeError(f"Missing staged logits for {sorted(missing)}")
    return output


def staged_probabilities(model: OPNsHybridClassifier, X, checkpoints: set[int]) -> dict[int, np.ndarray]:
    per_estimator = [
        _binary_staged_logits(estimator, X, checkpoints)
        for estimator in model.estimators_
    ]
    results: dict[int, np.ndarray] = {}
    for checkpoint in sorted(checkpoints):
        logits = [np.clip(values[checkpoint], -40.0, 40.0) for values in per_estimator]
        if len(logits) == 1:
            p = 1.0 / (1.0 + np.exp(-logits[0]))
            results[checkpoint] = np.column_stack([1.0 - p, p])
        else:
            probs = np.column_stack([1.0 / (1.0 + np.exp(-z)) for z in logits])
            sums = probs.sum(axis=1, keepdims=True)
            sums[sums <= 0] = 1.0
            results[checkpoint] = probs / sums
    return results


def classification_scores(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    pred = classes[np.argmax(proba, axis=1)]
    average = "binary" if len(classes) == 2 else "macro"
    scores = {
        "LogLoss": float(log_loss(y_true, proba, labels=classes)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, average=average, zero_division=0)),
    }
    try:
        if len(classes) == 2:
            scores["AUC"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            scores["AUC"] = float(
                roc_auc_score(
                    y_true,
                    proba,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        scores["AUC"] = math.nan
    return scores


def cumulative_sum(model: OPNsHybridClassifier, attribute: str, n_rounds: int) -> int:
    total = 0
    for estimator in model.estimators_:
        values = getattr(estimator, attribute, [])
        total += sum(int(v) for v in values[:n_rounds])
    return int(total)


def time_to_checkpoint(model: OPNsHybridClassifier, n_rounds: int) -> float:
    total = 0.0
    for estimator in model.estimators_:
        total += float(getattr(estimator, "phase1_time_", 0.0))
        total += float(getattr(estimator, "feature_pool_time_", 0.0))
        total += float(getattr(estimator, "threshold_cache_time_", 0.0))
        total += float(np.sum(getattr(estimator, "tree_iteration_times_", [])[:n_rounds]))
    return float(total)
