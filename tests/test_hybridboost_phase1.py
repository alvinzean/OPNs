from __future__ import annotations

import unittest

import numpy as np

import opns_pack.opns_np as op

from opns_boost._phase1 import Lasso, LogisticRegression
from opns_module.preprocessing import OPNsStandardScaler
from opns_pack.opns import OPNs


class TestHybridBoostPhase1(unittest.TestCase):
    @staticmethod
    def make_probe_data():
        rows = [
            [
                OPNs(0.10, -0.40),
                OPNs(0.30, 0.80),
            ],
            [
                OPNs(0.50, 0.20),
                OPNs(-0.10, 0.60),
            ],
            [
                OPNs(0.90, -0.30),
                OPNs(0.70, 0.10),
            ],
            [
                OPNs(-0.20, 0.50),
                OPNs(0.40, -0.70),
            ],
            [
                OPNs(0.60, 0.90),
                OPNs(-0.50, 0.20),
            ],
            [
                OPNs(-0.80, -0.10),
                OPNs(0.20, 0.50),
            ],
            [
                OPNs(0.40, -0.60),
                OPNs(0.80, -0.20),
            ],
            [
                OPNs(-0.30, 0.70),
                OPNs(-0.60, 0.40),
            ],
        ]

        target = np.asarray(
            [
                0.10,
                0.45,
                0.25,
                0.80,
                0.60,
                0.35,
                0.90,
                0.55,
            ],
            dtype=float,
        )

        X = op.array(rows)

        scaler = OPNsStandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, target

    def test_lasso_matches_frozen_behavior_probe(self) -> None:
        X, target = self.make_probe_data()

        model = Lasso(
            alpha=0.001,
            max_iter=8,
            tol=1.0e-10,
            learning_rate=0.01,
            adaptive_lr=False,
            adaptive_iter=False,
        )

        model.fit(
            X,
            target,
        )

        prediction_a = np.asarray(
            model.predict(
                X,
                item=0,
            ),
            dtype=float,
        ).reshape(-1)

        prediction_ab = np.asarray(
            model.predict(
                X,
                item=1,
            ),
            dtype=float,
        ).reshape(-1)

        expected_a = np.asarray(
            [
                0.3058859597606012,
                0.3654853014359778,
                0.38247681429321145,
                0.801988490967994,
                0.5240765881271661,
                0.522324576964066,
                0.5178478818786603,
                0.5799143865723234,
            ],
            dtype=float,
        )

        expected_ab = np.asarray(
            [
                0.2711653993121084,
                0.40287563111516633,
                0.29685605075565763,
                0.752535286170489,
                0.6320418927965499,
                0.4861596912977496,
                0.45455877732095284,
                0.7038072712313264,
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            prediction_a,
            expected_a,
            rtol=0.0,
            atol=5.0e-15,
        )

        np.testing.assert_allclose(
            prediction_ab,
            expected_ab,
            rtol=0.0,
            atol=5.0e-15,
        )

        self.assertEqual(
            model.stop_reason_,
            "max_iter",
        )

        self.assertEqual(
            model.best_iteration_,
            8,
        )

        self.assertTrue(
            hasattr(
                model,
                "max_coordinate_residual_",
            )
        )

        self.assertTrue(
            hasattr(
                model,
                "final_objective_",
            )
        )

    def test_logistic_regression_smoke(self) -> None:
        X, _ = self.make_probe_data()

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

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=8,
            adapt_lr=False,
            adapt_max_iter=False,
            tol=1.0e-10,
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
            probabilities.shape,
            (
                len(labels),
                2,
            ),
        )

        self.assertEqual(
            predictions.shape,
            labels.shape,
        )

        self.assertTrue(
            np.isfinite(
                probabilities
            ).all()
        )

        self.assertTrue(
            (
                probabilities
                >= -1.0e-12
            ).all()
        )

        self.assertTrue(
            (
                probabilities
                <= 1.0 + 1.0e-12
            ).all()
        )

        np.testing.assert_allclose(
            probabilities.sum(
                axis=1
            ),
            np.ones(
                len(labels)
            ),
            rtol=0.0,
            atol=1.0e-10,
        )

        self.assertTrue(
            set(
                predictions.tolist()
            ).issubset(
                {
                    0,
                    1,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
