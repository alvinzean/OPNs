from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import opns_boost.experiment as experiment


class FakeRegressor:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.fit_call = None
        FakeRegressor.instances.append(self)

    def fit(self, X, y, X_val=None, y_val=None):
        self.fit_call = {
            "X": X,
            "y": y,
            "X_val": X_val,
            "y_val": y_val,
        }
        return self

    def predict(self, X, item=1):
        self.predict_X = X
        self.predict_item = item
        return np.asarray([0.25, 0.75], dtype=float)

    def get_diagnostics(self):
        return {
            "active_features": [0, 1],
            "tree_search_features": 7,
            "diagnostic_marker": "regression-ok",
        }


class FakeClassifier:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.fit_call = None
        FakeClassifier.instances.append(self)

    def fit(self, X, y, X_val=None, y_val=None):
        self.fit_call = {
            "X": X,
            "y": y,
            "X_val": X_val,
            "y_val": y_val,
        }
        return self

    def predict(self, X):
        self.predict_X = X
        return np.asarray([1, 0], dtype=int)

    def predict_proba(self, X):
        self.proba_X = X
        return np.asarray(
            [
                [0.20, 0.80],
                [0.70, 0.30],
            ],
            dtype=float,
        )

    def get_diagnostics(self):
        return {
            "active_features": [0, 2],
            "classification_marker": "classification-ok",
        }


class TestHybridBoostExperiment(unittest.TestCase):
    def setUp(self) -> None:
        FakeRegressor.instances = []
        FakeClassifier.instances = []

    def test_resolve_dataset_params_contract(self) -> None:
        self.assertEqual(
            experiment.resolve_dataset_params(
                None,
                "wine_quality",
            ),
            {},
        )

        flat = {
            "n_estimators": 100,
            "max_depth": 4,
        }

        resolved_flat = experiment.resolve_dataset_params(
            flat,
            "wine_quality",
        )

        self.assertEqual(
            resolved_flat,
            flat,
        )

        self.assertIsNot(
            resolved_flat,
            flat,
        )

        nested = {
            "__default__": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
            },
            "wine_quality": {
                "max_depth": 7,
                "learning_rate": 0.2,
            },
        }

        resolved = experiment.resolve_dataset_params(
            nested,
            "wine_quality",
        )

        self.assertEqual(
            resolved,
            {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.2,
            },
        )

        self.assertEqual(
            nested["__default__"]["max_depth"],
            5,
        )

    def test_serialize_param_value_contract(self) -> None:
        self.assertEqual(
            experiment.serialize_param_value(None),
            "None",
        )

        self.assertEqual(
            experiment.serialize_param_value(3),
            3,
        )

        self.assertEqual(
            experiment.serialize_param_value(0.25),
            0.25,
        )

        self.assertEqual(
            experiment.serialize_param_value(True),
            True,
        )

        self.assertEqual(
            experiment.serialize_param_value(
                "fast_exact"
            ),
            "fast_exact",
        )

        self.assertEqual(
            experiment.serialize_param_value(
                [1, 2]
            ),
            "[1, 2]",
        )

    def test_regression_rejects_invalid_progress_interval(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "progress_interval must be a positive integer",
        ):
            experiment.run_opns_regression_cv(
                [],
                progress_interval=0,
            )

    def test_regression_cv_orchestration_contract(
        self,
    ) -> None:
        prepared = SimpleNamespace(
            X_opns=np.asarray(
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                ],
                dtype=object,
            ),
            y_model=np.asarray(
                [
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                ],
                dtype=object,
            ),
            candidate_pairs=5,
        )

        splits = [
            (
                np.asarray([0, 1], dtype=int),
                np.asarray([2, 3], dtype=int),
            )
        ]

        model_params = {
            "__default__": {
                "n_estimators": 12,
                "max_depth": 3,
            },
            "toy_reg": {
                "max_depth": 6,
            },
        }

        summary_sentinel = pd.DataFrame(
            [{"rows": 1}]
        )

        with (
            patch.object(
                experiment,
                "prepare_regression_data",
                return_value=prepared,
            ) as prepare_mock,
            patch.object(
                experiment,
                "repeated_splits",
                return_value=iter(splits),
            ) as split_mock,
            patch.object(
                experiment,
                "extract_real",
                side_effect=lambda values, item=1: (
                    np.asarray(values, dtype=float)
                ),
            ),
            patch.object(
                experiment,
                "OPNsHybridRegressor",
                FakeRegressor,
            ),
            patch.object(
                experiment,
                "regression_metrics",
                return_value={
                    "RMSE": 0.125,
                    "MAE": 0.100,
                    "R2": 0.900,
                },
            ) as metrics_mock,
            patch.object(
                experiment,
                "summarize_records",
                return_value=summary_sentinel,
            ) as summary_mock,
            patch.object(
                experiment.time,
                "perf_counter",
                side_effect=[
                    0.0,
                    10.0,
                    13.0,
                    20.0,
                    20.5,
                ],
            ),
        ):
            raw, summary = (
                experiment.run_opns_regression_cv(
                    ["toy_reg"],
                    model_params=model_params,
                    n_splits=2,
                    n_repeats=1,
                    random_state=101,
                    pairing="permutations",
                    verbose=False,
                    progress_interval=17,
                )
            )

        prepare_mock.assert_called_once_with(
            "toy_reg",
            pairing="permutations",
        )

        split_args = split_mock.call_args.args

        np.testing.assert_allclose(
            split_args[0],
            [
                0.10,
                0.20,
                0.30,
                0.40,
            ],
        )

        self.assertEqual(
            split_args[1:],
            (
                "regression",
                2,
                1,
                101,
            ),
        )

        self.assertEqual(
            len(FakeRegressor.instances),
            1,
        )

        model = FakeRegressor.instances[0]

        self.assertEqual(
            model.init_kwargs["n_estimators"],
            12,
        )

        self.assertEqual(
            model.init_kwargs["max_depth"],
            6,
        )

        self.assertEqual(
            model.init_kwargs["random_state"],
            101,
        )

        self.assertEqual(
            model.init_kwargs["progress_interval"],
            17,
        )

        self.assertFalse(
            model.init_kwargs["verbose"]
        )

        np.testing.assert_array_equal(
            model.fit_call["X"],
            prepared.X_opns[[0, 1]],
        )

        np.testing.assert_array_equal(
            model.fit_call["X_val"],
            prepared.X_opns[[2, 3]],
        )

        metric_args = metrics_mock.call_args.args

        np.testing.assert_allclose(
            metric_args[0],
            [0.30, 0.40],
        )

        np.testing.assert_allclose(
            metric_args[1],
            [0.25, 0.75],
        )

        self.assertEqual(
            len(raw),
            1,
        )

        record = raw.iloc[0].to_dict()

        self.assertEqual(
            record["dataset"],
            "toy_reg",
        )

        self.assertEqual(
            record["task_type"],
            "regression",
        )

        self.assertEqual(
            record["method"],
            "OPNs-HybridBoost",
        )

        self.assertEqual(
            record["seed"],
            101,
        )

        self.assertEqual(
            record["candidate_pairs"],
            5,
        )

        self.assertAlmostEqual(
            record["train_time"],
            3.0,
        )

        self.assertAlmostEqual(
            record["inference_time"],
            0.5,
        )

        self.assertEqual(
            record["diagnostic_marker"],
            "regression-ok",
        )

        self.assertNotIn(
            "active_features",
            record,
        )

        self.assertEqual(
            record["param_n_estimators"],
            12,
        )

        self.assertEqual(
            record["param_max_depth"],
            6,
        )

        self.assertEqual(
            record["param_progress_interval"],
            17,
        )

        self.assertEqual(
            record["param_learning_rate"],
            "None",
        )

        summary_mock.assert_called_once()

        self.assertIs(
            summary,
            summary_sentinel,
        )

    def test_classification_cv_orchestration_contract(
        self,
    ) -> None:
        prepared = SimpleNamespace(
            X_opns=np.asarray(
                [
                    [1, 10],
                    [2, 20],
                    [3, 30],
                    [4, 40],
                ],
                dtype=object,
            ),
            y_model=np.asarray(
                [0, 1, 1, 0],
                dtype=int,
            ),
            candidate_pairs=4,
        )

        splits = [
            (
                np.asarray([0, 1], dtype=int),
                np.asarray([2, 3], dtype=int),
            )
        ]

        model_params = {
            "__default__": {
                "n_estimators": 9,
                "max_depth": 2,
            },
            "toy_cls": {
                "max_depth": 4,
            },
        }

        summary_sentinel = pd.DataFrame(
            [{"rows": 1}]
        )

        with (
            patch.object(
                experiment,
                "prepare_classification_data",
                return_value=prepared,
            ) as prepare_mock,
            patch.object(
                experiment,
                "repeated_splits",
                return_value=iter(splits),
            ) as split_mock,
            patch.object(
                experiment,
                "OPNsHybridClassifier",
                FakeClassifier,
            ),
            patch.object(
                experiment,
                "classification_metrics",
                return_value={
                    "Accuracy": 0.75,
                    "F1": 0.70,
                },
            ) as metrics_mock,
            patch.object(
                experiment,
                "summarize_records",
                return_value=summary_sentinel,
            ) as summary_mock,
            patch.object(
                experiment.time,
                "perf_counter",
                side_effect=[
                    10.0,
                    12.5,
                    20.0,
                    20.25,
                ],
            ),
        ):
            raw, summary = (
                experiment.run_opns_classification_cv(
                    ["toy_cls"],
                    model_params=model_params,
                    n_splits=2,
                    n_repeats=1,
                    random_state=202,
                    pairing="permutations",
                    max_original_features=8,
                )
            )

        prepare_mock.assert_called_once_with(
            "toy_cls",
            pairing="permutations",
            max_original_features=8,
        )

        split_args = split_mock.call_args.args

        np.testing.assert_array_equal(
            split_args[0],
            prepared.y_model,
        )

        self.assertEqual(
            split_args[1:],
            (
                "classification",
                2,
                1,
                202,
            ),
        )

        self.assertEqual(
            len(FakeClassifier.instances),
            1,
        )

        model = FakeClassifier.instances[0]

        self.assertEqual(
            model.init_kwargs["n_estimators"],
            9,
        )

        self.assertEqual(
            model.init_kwargs["max_depth"],
            4,
        )

        self.assertEqual(
            model.init_kwargs["random_state"],
            202,
        )

        self.assertFalse(
            model.init_kwargs["verbose"]
        )

        metric_args = metrics_mock.call_args.args

        np.testing.assert_array_equal(
            metric_args[0],
            [1, 0],
        )

        np.testing.assert_array_equal(
            metric_args[1],
            [1, 0],
        )

        np.testing.assert_allclose(
            metric_args[2],
            [
                [0.20, 0.80],
                [0.70, 0.30],
            ],
        )

        self.assertEqual(
            len(raw),
            1,
        )

        record = raw.iloc[0].to_dict()

        self.assertEqual(
            record["dataset"],
            "toy_cls",
        )

        self.assertEqual(
            record["task_type"],
            "classification",
        )

        self.assertEqual(
            record["method"],
            "OPNs-HybridBoost",
        )

        self.assertEqual(
            record["seed"],
            202,
        )

        self.assertEqual(
            record["candidate_pairs"],
            4,
        )

        self.assertAlmostEqual(
            record["train_time"],
            2.5,
        )

        self.assertAlmostEqual(
            record["inference_time"],
            0.25,
        )

        self.assertEqual(
            record["classification_marker"],
            "classification-ok",
        )

        self.assertNotIn(
            "active_features",
            record,
        )

        self.assertEqual(
            record["param_n_estimators"],
            9,
        )

        self.assertEqual(
            record["param_max_depth"],
            4,
        )

        summary_mock.assert_called_once()

        self.assertIs(
            summary,
            summary_sentinel,
        )


if __name__ == "__main__":
    unittest.main()