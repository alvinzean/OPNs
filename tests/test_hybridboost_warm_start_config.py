from __future__ import annotations

import hashlib
import inspect
import json
import unittest

from pathlib import Path

from opns_boost.core import (
    OPNsHybridRegressor,
)
from opns_boost.experiment import (
    resolve_dataset_params,
)


ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_DIR = (
    ROOT
    / "research"
    / "hybridboost"
    / "configs"
)

REGRESSION_WARM_PATH = (
    CONFIG_DIR
    / "regression_warm_start_final.json"
)

REGRESSION_OVERALL_PATH = (
    CONFIG_DIR
    / "regression_final.json"
)

CLASSIFICATION_FINAL_PATH = (
    CONFIG_DIR
    / "classification_final.json"
)


REGRESSION_WARM_SHA256 = (
    "3845f36adafb14396ad171758c38eafa3"
    "8c27c2159d13c8aece3dc54cd8510c7"
)

CLASSIFICATION_FINAL_SHA256 = (
    "2f64a5e5e210da757e803124e2bd789f"
    "f20adc8ae2652089b5c492dca23c8f79"
)

FORMAL_REGRESSION_DATASETS = (
    "airfoil",
    "concrete",
    "energy_cooling",
    "wine_quality",
)


def sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(
    path: Path,
):
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


class TestHybridBoostWarmStartConfig(
    unittest.TestCase,
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.warm = load_json(
            REGRESSION_WARM_PATH
        )

        cls.overall = load_json(
            REGRESSION_OVERALL_PATH
        )

    def test_regression_warm_config_frozen_hash(
        self,
    ) -> None:
        self.assertEqual(
            sha256(
                REGRESSION_WARM_PATH
            ),
            REGRESSION_WARM_SHA256,
        )

    def test_regression_warm_config_constructor_contract(
        self,
    ) -> None:
        signature = inspect.signature(
            OPNsHybridRegressor.__init__
        )

        allowed = {
            name
            for name, parameter
            in signature.parameters.items()
            if (
                name != "self"
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            )
        }

        for section, values in (
            self.warm.items()
        ):
            with self.subTest(
                section=section,
            ):
                self.assertIsInstance(
                    values,
                    dict,
                )

                self.assertEqual(
                    set(
                        values
                    )
                    - allowed,
                    set(),
                )

        for dataset in (
            FORMAL_REGRESSION_DATASETS
        ):
            with self.subTest(
                dataset=dataset,
            ):
                resolved = (
                    resolve_dataset_params(
                        self.warm,
                        dataset,
                    )
                )

                self.assertIsInstance(
                    resolved,
                    dict,
                )

                self.assertTrue(
                    resolved
                )

    def test_public_warm_start_config_reuse_contract(
        self,
    ) -> None:
        # Regression warm-start uses a distinct frozen
        # parameter configuration from Overall.
        self.assertNotEqual(
            self.warm,
            self.overall,
        )

        # Classification warm-start does not need a
        # duplicate public config: the corrected final
        # classification configuration is already the
        # canonical formal configuration.
        self.assertEqual(
            sha256(
                CLASSIFICATION_FINAL_PATH
            ),
            CLASSIFICATION_FINAL_SHA256,
        )


if __name__ == "__main__":
    unittest.main()