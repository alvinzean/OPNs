from __future__ import annotations

import json
import math
import unittest

from pathlib import Path

import numpy as np
import pandas as pd

from opns_boost.metrics import (
    average_rank_table,
    classification_metrics,
    regression_metrics,
    summarize_records,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "metrics_frozen_probe.json"
)


def normalize_scalar(value):
    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        float,
    ):
        if math.isnan(
            value
        ):
            return "__NaN__"

        if math.isinf(
            value
        ):
            return (
                "__Inf__"
                if value > 0
                else "__NegInf__"
            )

        return float(
            value
        )

    if isinstance(
        value,
        (
            int,
            str,
            bool,
            type(None),
        ),
    ):
        return value

    return str(
        value
    )


def normalize_dict(value):
    return {
        str(key): normalize_scalar(
            item
        )
        for key, item
        in value.items()
    }


def normalize_frame(frame):
    return {
        "columns": [
            str(column)
            for column
            in frame.columns
        ],
        "index": [
            normalize_scalar(
                value
            )
            for value
            in frame.index.tolist()
        ],
        "records": [
            {
                str(key): normalize_scalar(
                    value
                )
                for key, value
                in record.items()
            }
            for record
            in frame.to_dict(
                orient="records"
            )
        ],
        "dtypes": {
            str(column): str(
                dtype
            )
            for column, dtype
            in frame.dtypes.items()
        },
    }


def normalize_rank(value):
    if isinstance(
        value,
        pd.DataFrame,
    ):
        return normalize_frame(
            value
        )

    if isinstance(
        value,
        pd.Series,
    ):
        return {
            "type": "Series",
            "name": value.name,
            "index": [
                normalize_scalar(
                    item
                )
                for item
                in value.index.tolist()
            ],
            "values": [
                normalize_scalar(
                    item
                )
                for item
                in value.tolist()
            ],
        }

    if value is None:
        return None

    return {
        "type": type(
            value
        ).__name__,
        "repr": repr(
            value
        ),
    }


class TestHybridBoostMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(
            FIXTURE_PATH.read_text(
                encoding="utf-8"
            )
        )[
            "behavior"
        ]

    def test_regression_metrics_matches_frozen_probe(
        self,
    ) -> None:
        y_true = np.asarray(
            [
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
            ],
            dtype=float,
        )

        y_pred = np.asarray(
            [
                1.2,
                1.7,
                4.5,
                7.0,
                15.5,
            ],
            dtype=float,
        )

        actual = normalize_dict(
            regression_metrics(
                y_true,
                y_pred,
            )
        )

        self.assertEqual(
            actual,
            self.expected[
                "regression_metrics"
            ],
        )

    def test_classification_metrics_matches_frozen_probe(
        self,
    ) -> None:
        y_true_binary = np.asarray(
            [
                0,
                0,
                1,
                1,
                0,
                1,
            ],
            dtype=int,
        )

        y_pred_binary = np.asarray(
            [
                0,
                1,
                1,
                1,
                0,
                0,
            ],
            dtype=int,
        )

        y_proba_binary = np.asarray(
            [
                [0.90, 0.10],
                [0.40, 0.60],
                [0.20, 0.80],
                [0.15, 0.85],
                [0.70, 0.30],
                [0.55, 0.45],
            ],
            dtype=float,
        )

        actual_binary = normalize_dict(
            classification_metrics(
                y_true_binary,
                y_pred_binary,
                y_proba_binary,
            )
        )

        self.assertEqual(
            actual_binary,
            self.expected[
                "classification_binary"
            ],
        )

        y_true_multi = np.asarray(
            [
                0,
                1,
                2,
                0,
                1,
                2,
            ],
            dtype=int,
        )

        y_pred_multi = np.asarray(
            [
                0,
                1,
                1,
                0,
                2,
                2,
            ],
            dtype=int,
        )

        y_proba_multi = np.asarray(
            [
                [0.80, 0.10, 0.10],
                [0.15, 0.70, 0.15],
                [0.10, 0.40, 0.50],
                [0.70, 0.20, 0.10],
                [0.15, 0.30, 0.55],
                [0.10, 0.20, 0.70],
            ],
            dtype=float,
        )

        actual_multi = normalize_dict(
            classification_metrics(
                y_true_multi,
                y_pred_multi,
                y_proba_multi,
            )
        )

        self.assertEqual(
            actual_multi,
            self.expected[
                "classification_multiclass"
            ],
        )

    def test_summarize_records_matches_frozen_probe(
        self,
    ) -> None:
        records = [
            {
                "dataset": "d1",
                "method": "M1",
                "task_type": "regression",
                "split": 0,
                "seed": 42,
                "RMSE": 1.0,
                "MAE": 0.8,
                "R2": 0.70,
                "train_time": 10.0,
                "param_max_depth": 4,
                "param_backend": "fast_exact",
            },
            {
                "dataset": "d1",
                "method": "M1",
                "task_type": "regression",
                "split": 1,
                "seed": 43,
                "RMSE": 1.2,
                "MAE": 0.9,
                "R2": 0.75,
                "train_time": 12.0,
                "param_max_depth": 4,
                "param_backend": "fast_exact",
            },
            {
                "dataset": "d1",
                "method": "M2",
                "task_type": "regression",
                "split": 0,
                "seed": 42,
                "RMSE": 0.9,
                "MAE": 0.7,
                "R2": 0.80,
                "train_time": 8.0,
                "param_max_depth": 5,
                "param_backend": "ordered_exact",
            },
            {
                "dataset": "d1",
                "method": "M2",
                "task_type": "regression",
                "split": 1,
                "seed": 43,
                "RMSE": 1.1,
                "MAE": 0.75,
                "R2": 0.78,
                "train_time": 9.0,
                "param_max_depth": 6,
                "param_backend": "ordered_exact",
            },
        ]

        self.assertEqual(
            normalize_frame(
                summarize_records(
                    records
                )
            ),
            self.expected[
                "summary_from_list"
            ],
        )

        self.assertEqual(
            normalize_frame(
                summarize_records(
                    pd.DataFrame(
                        records
                    )
                )
            ),
            self.expected[
                "summary_from_frame"
            ],
        )

        self.assertEqual(
            normalize_frame(
                summarize_records(
                    []
                )
            ),
            self.expected[
                "summary_empty"
            ],
        )

        try:
            summarize_records(
                [
                    {
                        "RMSE": 1.0,
                    }
                ]
            )

        except Exception as error:
            actual_error = {
                "type": type(
                    error
                ).__name__,
                "message": str(
                    error
                ),
            }

        else:
            actual_error = None

        self.assertEqual(
            actual_error,
            self.expected[
                "summary_missing_group_error"
            ],
        )

    def test_average_rank_table_matches_frozen_probe(
        self,
    ) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "d1",
                    "method": "M1",
                    "RMSE_mean": 1.0,
                },
                {
                    "dataset": "d1",
                    "method": "M2",
                    "RMSE_mean": 0.8,
                },
                {
                    "dataset": "d2",
                    "method": "M1",
                    "RMSE_mean": 0.7,
                },
                {
                    "dataset": "d2",
                    "method": "M2",
                    "RMSE_mean": 0.9,
                },
            ]
        )

        lower = average_rank_table(
            frame,
            metric="RMSE",
            higher_is_better=False,
        )

        higher = average_rank_table(
            frame.rename(
                columns={
                    "RMSE_mean": (
                        "Accuracy_mean"
                    )
                }
            ),
            metric="Accuracy",
            higher_is_better=True,
        )

        self.assertEqual(
            normalize_rank(
                lower
            ),
            self.expected[
                "average_rank_lower"
            ],
        )

        self.assertEqual(
            normalize_rank(
                higher
            ),
            self.expected[
                "average_rank_higher"
            ],
        )


if __name__ == "__main__":
    unittest.main()
