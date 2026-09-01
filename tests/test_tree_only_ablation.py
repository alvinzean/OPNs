from __future__ import annotations

import json
import unittest

from pathlib import Path

import numpy as np

from opns_boost.core import (
    OPNs,
    op,
)
from research.hybridboost.ablations.tree_only import (
    OPNsTreeOnlyRegressor,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tree_only_frozen_probe.json"
)


def pair(value):
    return [
        float(value.a),
        float(value.b),
    ]


def make_data():
    X = op.array(
        [
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
    )

    y = op.array(
        [
            OPNs(value, 0.0)
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

    return X, y


def tree_structure(model):
    structures = []

    for tree, lr in model.trees_:
        structures.append(
            {
                "lr": pair(lr),
                "splits": [
                    {
                        "feature": int(
                            feature
                        ),
                        "threshold": pair(
                            threshold
                        ),
                    }
                    for (
                        feature,
                        threshold,
                    )
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
        )

    return structures


class TestTreeOnlyAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(
            FIXTURE_PATH.read_text(
                encoding="utf-8"
            )
        )[
            "behavior"
        ]

    def test_research_location_and_constructor_guards(
        self,
    ) -> None:
        self.assertEqual(
            OPNsTreeOnlyRegressor.__module__,
            (
                "research.hybridboost."
                "ablations.tree_only"
            ),
        )

        expected = self.expected[
            "invalid_mode_error"
        ]

        with self.assertRaisesRegex(
            ValueError,
            expected[
                "message"
            ],
        ):
            OPNsTreeOnlyRegressor(
                warm_start_mode="lasso",
            )

        expected_feature = self.expected[
            "invalid_feature_mode_error"
        ]

        with self.assertRaisesRegex(
            ValueError,
            expected_feature[
                "message"
            ],
        ):
            OPNsTreeOnlyRegressor(
                warm_start_mode="constant",
                tree_feature_mode="active",
            )

    def test_matches_frozen_tree_only_probe(
        self,
    ) -> None:
        X, y = make_data()

        model = OPNsTreeOnlyRegressor(
            warm_start_mode="constant",
            n_estimators=2,
            learning_rate=0.1,
            max_depth=2,
            l2_leaf_reg=0.1,
            feature_filter_ratio=1.0,
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
            verbose=False,
            deliberately_ignored_parameter=123,
        )

        model.fit(
            X,
            y,
        )

        prediction = np.asarray(
            model.predict(
                X,
                item=1,
                backend="object_batch",
            ),
            dtype=float,
        )

        legacy = np.asarray(
            model.predict(
                X,
                item=1,
                backend="legacy_object",
            ),
            dtype=float,
        )

        spectral_label = np.asarray(
            model.predict(
                X,
                item=1,
                backend="spectral_batch",
            ),
            dtype=float,
        )

        np.testing.assert_allclose(
            prediction,
            np.asarray(
                self.expected[
                    "prediction"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=5.0e-13,
        )

        np.testing.assert_allclose(
            legacy,
            np.asarray(
                self.expected[
                    "prediction_legacy"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=5.0e-13,
        )

        np.testing.assert_allclose(
            spectral_label,
            np.asarray(
                self.expected[
                    "prediction_spectral_label"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=5.0e-13,
        )

        self.assertEqual(
            model.ignored_config_params_,
            self.expected[
                "ignored_config_params"
            ],
        )

        self.assertEqual(
            model.base_model_ is None,
            self.expected[
                "base_model_is_none"
            ],
        )

        self.assertEqual(
            model.base_scaler_ is None,
            self.expected[
                "base_scaler_is_none"
            ],
        )

        np.testing.assert_allclose(
            pair(
                model.base_constant_
            ),
            self.expected[
                "base_constant"
            ],
            rtol=0.0,
            atol=5.0e-13,
        )

        self.assertEqual(
            np.asarray(
                model.active_features_,
                dtype=int,
            ).tolist(),
            self.expected[
                "active_features"
            ],
        )

        diagnostics = (
            model.get_diagnostics()
        )

        stable_keys = (
            "candidate_features",
            "phase1_time",
            "feature_pool_time",
            "threshold_cache_features",
            "threshold_cache_candidates",
            "candidate_feature_evaluations",
            "threshold_evaluations",
            "selected_features",
            "sparsity_ratio",
            "warm_start_mode",
            "phase1_enabled",
            "tree_feature_mode",
            "tree_search_features",
            "active_plus_added_features",
            "split_score_backend",
            "threshold_backend",
            "threshold_sampling",
            "prediction_backend",
            "n_trees",
            "best_iteration",
            "active_features",
        )

        stable_diagnostics = {
            key: diagnostics[key]
            for key in stable_keys
        }

        self.assertEqual(
            stable_diagnostics,
            self.expected[
                "stable_diagnostics"
            ],
        )

        self.assertEqual(
            model.prediction_equivalence_report(
                X
            ),
            self.expected[
                "prediction_equivalence_report"
            ],
        )

        self.assertEqual(
            tree_structure(
                model
            ),
            self.expected[
                "trees"
            ],
        )


if __name__ == "__main__":
    unittest.main()
