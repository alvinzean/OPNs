from __future__ import annotations

import json
import unittest

from pathlib import Path

import numpy as np

import opns_pack.opns_np as op

from opns_boost.core import (
    OPNsHybridClassifier,
    OPNsHybridRegressor,
    OPNsObliviousTree,
)
from opns_pack.opns import OPNs


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "hybridboost_core_frozen_probe.json"
)


def load_fixture():
    return json.loads(
        FIXTURE_PATH.read_text(
            encoding="utf-8"
        )
    )


def make_features():
    raw_rows = [
        [(0.10, -0.40), (0.30, 0.80)],
        [(0.50, 0.20), (-0.10, 0.60)],
        [(0.90, -0.30), (0.70, 0.10)],
        [(-0.20, 0.50), (0.40, -0.70)],
        [(0.60, 0.90), (-0.50, 0.20)],
        [(-0.80, -0.10), (0.20, 0.50)],
        [(0.40, -0.60), (0.80, -0.20)],
        [(-0.30, 0.70), (-0.60, 0.40)],
    ]

    return op.array(
        [
            [
                OPNs(a, b)
                for a, b in row
            ]
            for row in raw_rows
        ]
    )


def pair(value):
    return [
        float(value.a),
        float(value.b),
    ]


def matrix_pairs(values):
    return [
        pair(value)
        for value in values.flatten()
    ]


def tree_structure(tree):
    return {
        "splits": [
            {
                "feature": int(feature),
                "threshold": pair(
                    threshold
                ),
            }
            for feature, threshold
            in tree.splits
        ],
        "leaves": {
            str(key): pair(value)
            for key, value
            in sorted(
                tree.leaf_values.items()
            )
        },
    }


class TestPredictionEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()
        cls.tolerance = float(
            cls.fixture[
                "source"
            ][
                "tolerance"
            ]
        )

    def test_threshold_candidate_equivalence(self) -> None:
        expected = self.fixture[
            "threshold_probe"
        ]

        column = op.array(
            [
                OPNs(1.0, 2.0),
                OPNs(2.0, 1.0),
                OPNs(0.0, 0.0),
                OPNs(3.0, -1.0),
                OPNs(-1.0, 4.0),
                OPNs(1.0, 2.0),
                OPNs(-2.0, -3.0),
            ]
        )

        tree = OPNsObliviousTree(
            max_depth=2,
            max_thresholds=3,
            random_strength=0.0,
            split_score_backend="fast_exact",
            threshold_backend="ordered_exact",
            threshold_sampling="stride",
            random_state=7,
        )

        unique = matrix_pairs(
            tree._unique_thresholds(
                column
            )
        )

        sampled = matrix_pairs(
            tree._thresholds(
                column
            )
        )

        constant_column = op.array(
            [
                OPNs(1.0, 2.0)
                for _ in range(8)
            ]
        )

        self.assertEqual(
            unique,
            expected["unique"],
        )

        self.assertEqual(
            sampled,
            expected["sampled"],
        )

        self.assertEqual(
            len(
                tree._thresholds(
                    constant_column
                )
            ),
            expected[
                "constant_candidate_count"
            ],
        )

    def test_regression_matches_frozen_probe(self) -> None:
        expected = self.fixture[
            "regression"
        ]

        X = make_features()

        target = op.array(
            [
                OPNs(float(value), 0.0)
                for value in (
                    0.10,
                    0.45,
                    0.25,
                    0.80,
                    0.60,
                    0.35,
                    0.90,
                    0.55,
                )
            ]
        )

        model = OPNsHybridRegressor(
            n_estimators=2,
            learning_rate=0.1,
            max_depth=2,
            l2_leaf_reg=0.1,
            poly_degree=1,
            use_trig=False,
            lasso_alpha=0.001,
            lasso_max_iter=8,
            feature_filter_ratio=1.0,
            active_threshold=1.0e-10,
            random_strength=0.0,
            lr_decay_type="constant",
            early_stopping_rounds=None,
            max_thresholds=3,
            tree_feature_mode="all",
            split_score_backend="fast_exact",
            threshold_backend="ordered_exact",
            threshold_sampling="stride",
            prediction_backend="object_batch",
            random_state=7,
        )

        model.fit(
            X,
            target,
        )

        prediction = np.asarray(
            model.predict(
                X,
                item=1,
            ),
            dtype=float,
        ).reshape(-1)

        np.testing.assert_allclose(
            prediction,
            np.asarray(
                expected[
                    "prediction"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=self.tolerance,
        )

        self.assertEqual(
            [
                int(value)
                for value
                in np.asarray(
                    model.active_features_
                ).reshape(-1)
            ],
            expected[
                "active_features"
            ],
        )

        self.assertEqual(
            getattr(
                model.base_model_,
                "stop_reason_",
                None,
            ),
            expected[
                "lasso_stop_reason"
            ],
        )

        self.assertEqual(
            getattr(
                model.base_model_,
                "best_iteration_",
                None,
            ),
            expected[
                "lasso_best_iteration"
            ],
        )

        self.assertEqual(
            [
                tree_structure(tree)
                for tree, _
                in model.trees_
            ],
            expected[
                "trees"
            ],
        )

    def test_classification_matches_frozen_probe(self) -> None:
        expected = self.fixture[
            "classification"
        ]

        X = make_features()

        labels = np.asarray(
            [
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                1,
            ],
            dtype=int,
        )

        model = OPNsHybridClassifier(
            n_estimators=1,
            learning_rate=0.1,
            max_depth=1,
            l2_leaf_reg=1.0,
            poly_degree=1,
            use_trig=False,
            logistic_lr=0.1,
            logistic_max_iter=8,
            feature_filter_ratio=1.0,
            active_threshold=1.0e-10,
            random_strength=0.0,
            lr_decay_type="constant",
            early_stopping_rounds=None,
            max_thresholds=3,
            split_score_backend="fast_exact",
            threshold_backend="ordered_exact",
            threshold_sampling="stride",
            random_state=7,
            phase1_enabled=True,
            tree_feature_mode="all",
        )

        model.fit(
            X,
            labels,
        )

        probabilities = np.asarray(
            model.predict_proba(X),
            dtype=float,
        )

        predictions = np.asarray(
            model.predict(X),
            dtype=int,
        ).reshape(-1)

        self.assertEqual(
            list(
                probabilities.shape
            ),
            expected[
                "probability_shape"
            ],
        )

        np.testing.assert_allclose(
            probabilities.reshape(-1),
            np.asarray(
                expected[
                    "probabilities"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=self.tolerance,
        )

        self.assertEqual(
            predictions.tolist(),
            expected[
                "predictions"
            ],
        )

        self.assertEqual(
            [
                [
                    int(value)
                    for value
                    in np.asarray(
                        estimator.active_features_
                    ).reshape(-1)
                ]
                for estimator
                in model.estimators_
            ],
            expected[
                "active_features"
            ],
        )

        self.assertEqual(
            [
                [
                    tree_structure(tree)
                    for tree, _
                    in estimator.trees_
                ]
                for estimator
                in model.estimators_
            ],
            expected[
                "trees"
            ],
        )


if __name__ == "__main__":
    unittest.main()
