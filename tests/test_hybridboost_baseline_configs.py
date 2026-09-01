from __future__ import annotations

import hashlib
import json
import unittest

from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_DIR = (
    ROOT
    / "research"
    / "hybridboost"
    / "configs"
)

CLASSIFICATION_PATH = (
    CONFIG_DIR
    / "classification_baselines_final.json"
)

REGRESSION_PATH = (
    CONFIG_DIR
    / "regression_baselines_final.json"
)


CLASSIFICATION_SHA256 = (
    "8b930dd4dd04df44886f8886e3f0cc22"
    "2b71b0c39e37efce7eef340bc9f78890"
)

REGRESSION_SHA256 = (
    "f9d2368b4e2d1494e38e62fda5e371e"
    "acc3208c4f08db327feaf159e5f576e90"
)


CLASSIFICATION_DATASETS = {
    "iris",
    "wine",
    "car",
    "breast_cancer",
}

REGRESSION_DATASETS = {
    "airfoil",
    "boston",
    "concrete",
    "energy_cooling",
    "wine_quality",
}

BASELINE_METHODS = {
    "xgboost",
    "lightgbm",
    "catboost",
    "histgb",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def dataset_keys(config):
    return {
        key
        for key in config
        if key != "__default__"
    }


def method_keys(block):
    return {
        key
        for key in block
        if key != "__default__"
    }


class TestHybridBoostBaselineConfigs(
    unittest.TestCase,
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classification = load_json(
            CLASSIFICATION_PATH
        )

        cls.regression = load_json(
            REGRESSION_PATH
        )

    def test_frozen_baseline_config_hashes(
        self,
    ) -> None:
        self.assertEqual(
            sha256(
                CLASSIFICATION_PATH
            ),
            CLASSIFICATION_SHA256,
        )

        self.assertEqual(
            sha256(
                REGRESSION_PATH
            ),
            REGRESSION_SHA256,
        )

    def test_dataset_and_method_coverage(
        self,
    ) -> None:
        self.assertEqual(
            dataset_keys(
                self.classification
            ),
            CLASSIFICATION_DATASETS,
        )

        self.assertEqual(
            dataset_keys(
                self.regression
            ),
            REGRESSION_DATASETS,
        )

        for dataset in (
            CLASSIFICATION_DATASETS
        ):
            with self.subTest(
                task="classification",
                dataset=dataset,
            ):
                self.assertEqual(
                    method_keys(
                        self.classification[
                            dataset
                        ]
                    ),
                    BASELINE_METHODS,
                )

        for dataset in (
            REGRESSION_DATASETS
        ):
            with self.subTest(
                task="regression",
                dataset=dataset,
            ):
                self.assertEqual(
                    method_keys(
                        self.regression[
                            dataset
                        ]
                    ),
                    BASELINE_METHODS,
                )

    def test_corrected_car_baseline_contract(
        self,
    ) -> None:
        car = self.classification[
            "car"
        ]

        self.assertEqual(
            car[
                "xgboost"
            ],
            {
                "n_estimators": 160,
                "learning_rate": 0.1,
                "max_depth": 4,
                "reg_lambda": 1.0,
            },
        )

        self.assertEqual(
            car[
                "lightgbm"
            ],
            {
                "n_estimators": 160,
                "learning_rate": 0.1,
                "max_depth": 4,
                "num_leaves": 16,
                "reg_lambda": 1.0,
            },
        )

        self.assertEqual(
            car[
                "catboost"
            ],
            {
                "iterations": 160,
                "learning_rate": 0.1,
                "depth": 4,
                "l2_leaf_reg": 1.0,
            },
        )

        self.assertEqual(
            car[
                "histgb"
            ],
            {
                "max_iter": 160,
                "learning_rate": 0.1,
                "max_depth": 4,
                "max_leaf_nodes": 16,
                "l2_regularization": 1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()