from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

import pandas as pd

from research.hybridboost.scripts import reproduce_active


class TestReproduceActive(
    unittest.TestCase,
):
    def test_default_efficiency_protocol_matches_formal_contract(
        self,
    ) -> None:
        args = reproduce_active.parse_args(
            [
                "--dry-run",
            ]
        )

        protocol = reproduce_active.build_protocol(
            args
        )

        self.assertEqual(
            args.experiment,
            "efficiency",
        )
        self.assertEqual(
            args.datasets,
            [
                "energy_cooling",
                "wine_quality",
            ],
        )
        self.assertEqual(
            protocol["protocol_name"],
            "active_efficiency",
        )
        self.assertEqual(
            protocol["modes"],
            [
                "full_all",
                "full_active",
            ],
        )
        self.assertEqual(
            protocol["n_splits"],
            5,
        )
        self.assertEqual(
            protocol["n_repeats"],
            1,
        )
        self.assertEqual(
            protocol["random_state"],
            161803,
        )
        self.assertEqual(
            protocol["checkpoint_step"],
            10,
        )
        self.assertIsNone(
            protocol["tree_budget_cap"]
        )
        self.assertTrue(
            protocol["raw_curve_is_primary"]
        )
        self.assertEqual(
            reproduce_active.SUMMARY_DDOF,
            1,
        )

    def test_structure_protocol_is_phase1_snapshot(
        self,
    ) -> None:
        args = reproduce_active.parse_args(
            [
                "--experiment",
                "structure",
                "--dry-run",
            ]
        )

        protocol = reproduce_active.build_protocol(
            args
        )

        self.assertEqual(
            args.datasets,
            [
                "airfoil",
                "concrete",
                "energy_cooling",
                "wine_quality",
            ],
        )
        self.assertEqual(
            protocol["protocol_name"],
            "phase1_structure_snapshot",
        )
        self.assertEqual(
            protocol["modes"],
            [
                "full_active",
            ],
        )
        self.assertEqual(
            protocol["tree_budget_cap"],
            0,
        )
        self.assertTrue(
            protocol["phase1_enabled"]
        )
        self.assertEqual(
            protocol["tree_feature_mode"],
            "active",
        )
        self.assertFalse(
            protocol["raw_curve_is_primary"]
        )
        self.assertIn(
            "not a boosting learning curve",
            protocol["interpretation"],
        )

    def test_frozen_regression_config_contract(
        self,
    ) -> None:
        actual = reproduce_active.sha256(
            reproduce_active.DEFAULT_REGRESSION_CONFIG
        )

        self.assertEqual(
            actual,
            (
                "3845f36adafb14396ad171758c38eafa3"
                "8c27c2159d13c8aece3dc54cd8510c7"
            ),
        )

        args = reproduce_active.parse_args(
            [
                "--dry-run",
            ]
        )

        protocol = reproduce_active.build_protocol(
            args
        )

        self.assertTrue(
            protocol["config"][
                "matches_expected"
            ]
        )

    def test_active_pairwise_ratios_and_sample_sd(
        self,
    ) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "demo",
                    "split": 0,
                    "n_trees": 10,
                    "mode": "full_all",
                    "status": "ok",
                    "RMSE": 2.0,
                    "MAE": 1.5,
                    "R2": 0.80,
                    "time_to_checkpoint": 10.0,
                    "tree_search_features": 20.0,
                    "active_retention_ratio": 1.0,
                    "candidate_feature_evaluations_cumulative": 100,
                    "threshold_evaluations_cumulative": 200,
                },
                {
                    "dataset": "demo",
                    "split": 0,
                    "n_trees": 10,
                    "mode": "full_active",
                    "status": "ok",
                    "RMSE": 2.1,
                    "MAE": 1.6,
                    "R2": 0.79,
                    "time_to_checkpoint": 6.0,
                    "tree_search_features": 12.0,
                    "active_retention_ratio": 0.60,
                    "candidate_feature_evaluations_cumulative": 60,
                    "threshold_evaluations_cumulative": 100,
                },
                {
                    "dataset": "demo",
                    "split": 1,
                    "n_trees": 10,
                    "mode": "full_all",
                    "status": "ok",
                    "RMSE": 2.2,
                    "MAE": 1.7,
                    "R2": 0.75,
                    "time_to_checkpoint": 12.0,
                    "tree_search_features": 20.0,
                    "active_retention_ratio": 1.0,
                    "candidate_feature_evaluations_cumulative": 120,
                    "threshold_evaluations_cumulative": 240,
                },
                {
                    "dataset": "demo",
                    "split": 1,
                    "n_trees": 10,
                    "mode": "full_active",
                    "status": "ok",
                    "RMSE": 2.3,
                    "MAE": 1.8,
                    "R2": 0.74,
                    "time_to_checkpoint": 6.0,
                    "tree_search_features": 10.0,
                    "active_retention_ratio": 0.50,
                    "candidate_feature_evaluations_cumulative": 60,
                    "threshold_evaluations_cumulative": 120,
                },
            ]
        )

        paired = (
            reproduce_active
            ._active_pairwise_curve(
                raw
            )
        )

        self.assertEqual(
            len(paired),
            2,
        )

        first = paired.iloc[0]

        self.assertAlmostEqual(
            first["active_retention"],
            0.60,
        )
        self.assertAlmostEqual(
            first["candidate_eval_ratio"],
            0.60,
        )
        self.assertAlmostEqual(
            first["threshold_eval_ratio"],
            0.50,
        )
        self.assertAlmostEqual(
            first["time_ratio"],
            0.60,
        )
        self.assertAlmostEqual(
            first[
                "RMSE_delta_active_minus_all"
            ],
            0.10,
        )

        summary = (
            reproduce_active
            ._active_pairwise_summary(
                paired
            )
        )

        self.assertEqual(
            int(
                summary.loc[
                    0,
                    "folds",
                ]
            ),
            2,
        )
        self.assertAlmostEqual(
            summary.loc[
                0,
                "active_retention_mean",
            ],
            0.55,
        )

        expected_std = pd.Series(
            [
                0.60,
                0.50,
            ]
        ).std(
            ddof=1
        )

        self.assertAlmostEqual(
            summary.loc[
                0,
                "active_retention_std",
            ],
            expected_std,
        )

    def test_dry_run_writes_metadata_and_config_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "active"

            status = reproduce_active.main(
                [
                    "--dry-run",
                    "--out-dir",
                    str(out),
                ]
            )

            self.assertEqual(
                status,
                0,
            )
            self.assertTrue(
                (
                    out
                    / "protocol.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    out
                    / "environment.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    out
                    / "regression_warm_start_final.json"
                ).is_file()
            )

            protocol = json.loads(
                (
                    out
                    / "protocol.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                protocol[
                    "protocol_name"
                ],
                "active_efficiency",
            )


if __name__ == "__main__":
    unittest.main()
