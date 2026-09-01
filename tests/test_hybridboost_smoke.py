from __future__ import annotations

import unittest

import numpy as np

import opns_pack.opns_np as op

from opns_boost.core import (
    OPNsHybridClassifier,
    OPNsHybridRegressor,
)
from opns_pack.opns import OPNs


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


class TestHybridBoostSmoke(unittest.TestCase):
    def test_regression_fit_predict(self) -> None:
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

        self.assertEqual(
            prediction.shape,
            (8,),
        )

        self.assertTrue(
            np.isfinite(
                prediction
            ).all()
        )

        self.assertGreater(
            len(model.trees_),
            0,
        )

    def test_classification_fit_predict(self) -> None:
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

        prediction = np.asarray(
            model.predict(X),
            dtype=int,
        )

        self.assertEqual(
            probabilities.shape,
            (8, 2),
        )

        self.assertEqual(
            prediction.shape,
            (8,),
        )

        self.assertTrue(
            np.isfinite(
                probabilities
            ).all()
        )

        self.assertTrue(
            set(
                prediction.tolist()
            ).issubset(
                {0, 1}
            )
        )


if __name__ == "__main__":
    unittest.main()
