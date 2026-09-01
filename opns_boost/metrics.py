"""Metrics and result aggregation utilities."""
from __future__ import annotations

from typing import Iterable, Any

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    multiclass = len(labels) > 2
    avg = "macro" if multiclass else "binary"
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            out["LogLoss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            out["LogLoss"] = np.nan
        try:
            if multiclass:
                out["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            else:
                p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["AUC"] = float(roc_auc_score(y_true, p))
        except ValueError:
            out["AUC"] = np.nan
    return out


def summarize_records(records: Iterable[dict] | pd.DataFrame, group_cols=("dataset", "method")) -> pd.DataFrame:
    """Summarize raw experiment records as mean ± std.

    Accepts either:
    1. a list of dict records; or
    2. an already constructed pandas DataFrame.
    """
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(list(records))

    if df.empty:
        return pd.DataFrame()

    missing_cols = [c for c in group_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Cannot summarize records because grouping columns are missing: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # excluded = set(group_cols) | {"split", "seed", "task_type"}
    # metric_cols = [
    #     c for c in df.columns
    #     if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    # ]
    excluded = set(group_cols) | {"split", "seed", "task_type"}

    metric_cols = [
        c for c in df.columns
        if c not in excluded
           and not c.startswith("param_")
           and pd.api.types.is_numeric_dtype(df[c])
    ]

    # rows = []
    # for keys, g in df.groupby(list(group_cols)):
    #     if not isinstance(keys, tuple):
    #         keys = (keys,)
    #
    #     row = dict(zip(group_cols, keys))
    #
    #     for col in metric_cols:
    #         mean = g[col].mean()
    #         std = g[col].std(ddof=1)
    #
    #         row[f"{col}_mean"] = mean
    #         row[f"{col}_std"] = std
    #         row[f"{col}_fmt"] = f"{mean:.4f} ± {std:.4f}"
    #
    #     rows.append(row)
    rows = []
    param_cols = [c for c in df.columns if c.startswith("param_")]

    for keys, g in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        # Keep per-dataset parameters in summary without mean/std formatting.
        for col in param_cols:
            unique_values = g[col].dropna().unique()
            if len(unique_values) == 1:
                row[col] = unique_values[0]
            elif len(unique_values) > 1:
                row[col] = ";".join(map(str, unique_values[:5]))

        for col in metric_cols:
            mean = g[col].mean()
            std = g[col].std(ddof=1)

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_fmt"] = f"{mean:.4f} ± {std:.4f}"

        rows.append(row)

    return pd.DataFrame(rows)


def average_rank_table(summary: pd.DataFrame, metric: str, higher_is_better: bool, method_col="method", dataset_col="dataset") -> \
Series[Any] | None:
    metric_col = f"{metric}_mean" if f"{metric}_mean" in summary.columns else metric
    rows = []
    for dataset, g in summary.groupby(dataset_col):
        ranks = g[metric_col].rank(ascending=not higher_is_better, method="average")
        for method, rank in zip(g[method_col], ranks):
            rows.append({"dataset": dataset, "method": method, "rank": float(rank)})
    rank_df = pd.DataFrame(rows)
    return rank_df.groupby("method", as_index=False)["rank"].mean().sort_values("rank")
