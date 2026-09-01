from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

import pandas as pd

from research.hybridboost.scripts import (
    reproduce_warm_start,
)


class TestReproduceWarmStart(
    unittest.TestCase,
):
    def test_default_protocol_matches_formal_contract(
        self,
    ) -> None:
        args = (
            reproduce_warm_start
            .parse_args(
                [
                    "--dry-run",
                ]
            )
        )

        self.assertEqual(
            args.classification_datasets,
            [
                "breast_cancer",
                "wine",
                "car",
                "iris",
            ],
        )

        self.assertEqual(
            args.regression_datasets,
            [
                "airfoil",
                "concrete",
                "energy_cooling",
                "wine_quality",
            ],
        )

        self.assertEqual(
            args.modes,
            [
                "full_all",
                "tree_only_all",
            ],
        )

        self.assertEqual(
            args.n_splits,
            5,
        )

        self.assertEqual(
            args.n_repeats,
            1,
        )

        self.assertEqual(
            args.random_state,
            161803,
        )

        self.assertEqual(
            args.checkpoint_step,
            10,
        )

        self.assertEqual(
            args.classification_round_budget_cap,
            60,
        )

        self.assertEqual(
            reproduce_warm_start
            .SUMMARY_DDOF,
            1,
        )

    def test_frozen_config_contract(
        self,
    ) -> None:
        self.assertEqual(
            reproduce_warm_start.sha256(
                reproduce_warm_start
                .DEFAULT_CLASSIFICATION_CONFIG
            ),
            (
                "2f64a5e5e210da757e803124e2bd789ff"
                "20adc8ae2652089b5c492dca23c8f79"
            ),
        )

        self.assertEqual(
            reproduce_warm_start.sha256(
                reproduce_warm_start
                .DEFAULT_REGRESSION_CONFIG
            ),
            (
                "3845f36adafb14396ad171758c38eafa3"
                "8c27c2159d13c8aece3dc54cd8510c7"
            ),
        )

        config = (
            reproduce_warm_start
            .load_json(
                reproduce_warm_start
                .DEFAULT_CLASSIFICATION_CONFIG
            )
        )

        car = (
            reproduce_warm_start
            .resolve_classification_params(
                config,
                "car",
                60,
            )
        )

        self.assertEqual(
            car[
                "n_estimators"
            ],
            60,
        )

    def test_wine_fold_local_policy(
        self,
    ) -> None:
        self.assertTrue(
            reproduce_warm_start
            .uses_fold_local_imputation(
                "wine_quality"
            )
        )

        self.assertFalse(
            reproduce_warm_start
            .uses_fold_local_imputation(
                "energy_cooling"
            )
        )

    def test_paired_gain_and_sample_sd_contract(
        self,
    ) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "demo",
                    "split": 0,
                    "mode": "full_all",
                    "status": "ok",
                    "n_trees": 10,
                    "RMSE": 1.0,
                    "R2": 0.8,
                    "time_to_checkpoint": 2.0,
                },
                {
                    "dataset": "demo",
                    "split": 0,
                    "mode": "tree_only_all",
                    "status": "ok",
                    "n_trees": 10,
                    "RMSE": 1.2,
                    "R2": 0.7,
                    "time_to_checkpoint": 1.5,
                },
                {
                    "dataset": "demo",
                    "split": 1,
                    "mode": "full_all",
                    "status": "ok",
                    "n_trees": 10,
                    "RMSE": 0.9,
                    "R2": 0.9,
                    "time_to_checkpoint": 2.2,
                },
                {
                    "dataset": "demo",
                    "split": 1,
                    "mode": "tree_only_all",
                    "status": "ok",
                    "n_trees": 10,
                    "RMSE": 1.0,
                    "R2": 0.8,
                    "time_to_checkpoint": 1.7,
                },
            ]
        )

        paired = (
            reproduce_warm_start
            ._paired_curve(
                raw,
                "n_trees",
                [
                    "RMSE",
                    "R2",
                ],
            )
        )

        self.assertAlmostEqual(
            paired.loc[
                0,
                "gain_RMSE",
            ],
            0.2,
        )

        self.assertAlmostEqual(
            paired.loc[
                1,
                "gain_RMSE",
            ],
            0.1,
        )

        self.assertAlmostEqual(
            paired.loc[
                0,
                "gain_R2",
            ],
            0.1,
        )

        self.assertAlmostEqual(
            paired.loc[
                1,
                "gain_R2",
            ],
            0.1,
        )

        summary = (
            reproduce_warm_start
            ._paired_summary(
                paired,
                "n_trees",
            )
        )

        expected_std = pd.Series(
            [
                0.2,
                0.1,
            ]
        ).std(
            ddof=1
        )

        self.assertAlmostEqual(
            summary.loc[
                0,
                "gain_RMSE_std",
            ],
            expected_std,
        )

        self.assertAlmostEqual(
            summary.loc[
                0,
                "extra_time_full_mean",
            ],
            0.5,
        )

    def test_dry_run_writes_metadata_and_snapshots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(
                tmp
            )

            status = (
                reproduce_warm_start
                .main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(
                            out
                        ),
                    ]
                )
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
                    / "classification_final.json"
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
                    "classification"
                ][
                    "tree_only_all"
                ][
                    "phase1_enabled"
                ],
                False,
            )

            self.assertEqual(
                protocol[
                    "regression"
                ][
                    "tree_only_all"
                ][
                    "warm_start_mode"
                ],
                "constant",
            )

            self.assertEqual(
                protocol[
                    "regression"
                ][
                    "prediction_backend"
                ],
                "object_batch",
            )


if __name__ == "__main__":
    unittest.main()