from __future__ import annotations

import hashlib
import inspect
import json
import unittest

from pathlib import Path

from opns_boost.core import (
    OPNsHybridClassifier,
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

CLASSIFICATION_PATH = (
    CONFIG_DIR
    / "classification_final.json"
)

REGRESSION_PATH = (
    CONFIG_DIR
    / "regression_final.json"
)


CLASSIFICATION_SHA256 = (
    "feafdb52b8c2e3debf5c64a21d0c3f1"
    "ee3c9f6635c755dfa1d547258bc4f7976"
)

REGRESSION_SHA256 = (
    "6c54747483d23a8b98e90861e6c920bd"
    "977d2e7fb38c04247f78486f456b0530"
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
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def constructor_parameters(
    cls,
) -> set[str]:
    signature = inspect.signature(
        cls.__init__
    )

    return {
        name
        for name, parameter
        in signature.parameters.items()
        if name != "self"
        and parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
    }


class TestHybridBoostFrozenConfigs(
    unittest.TestCase,
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.classification = load_json(
            CLASSIFICATION_PATH
        )

        cls.regression = load_json(
            REGRESSION_PATH
        )

    def test_frozen_config_hashes(
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

    def test_all_config_parameters_match_model_contracts(
        self,
    ) -> None:
        classifier_parameters = (
            constructor_parameters(
                OPNsHybridClassifier
            )
        )

        regressor_parameters = (
            constructor_parameters(
                OPNsHybridRegressor
            )
        )

        classification_unknown = {}

        for section, values in (
            self.classification.items()
        ):
            self.assertIsInstance(
                values,
                dict,
            )

            unknown = sorted(
                set(
                    values
                )
                - classifier_parameters
            )

            if unknown:
                classification_unknown[
                    section
                ] = unknown

        regression_unknown = {}

        for section, values in (
            self.regression.items()
        ):
            self.assertIsInstance(
                values,
                dict,
            )

            unknown = sorted(
                set(
                    values
                )
                - regressor_parameters
            )

            if unknown:
                regression_unknown[
                    section
                ] = unknown

        self.assertEqual(
            classification_unknown,
            {},
        )

        self.assertEqual(
            regression_unknown,
            {},
        )

    def test_dataset_override_resolution_contract(
        self,
    ) -> None:
        self.assertEqual(
            [
                key
                for key
                in self.classification
                if key != "__default__"
            ],
            [
                "breast_cancer",
                "car",
                "iris",
                "wine",
            ],
        )

        self.assertEqual(
            [
                key
                for key
                in self.regression
                if key != "__default__"
            ],
            [
                "airfoil",
                "boston",
                "concrete",
                "energy_cooling",
                "wine_quality",
            ],
        )

        classification_car_expected = dict(
            self.classification[
                "__default__"
            ]
        )

        classification_car_expected.update(
            self.classification[
                "car"
            ]
        )

        self.assertEqual(
            resolve_dataset_params(
                self.classification,
                "car",
            ),
            classification_car_expected,
        )

        regression_wine_expected = dict(
            self.regression[
                "__default__"
            ]
        )

        regression_wine_expected.update(
            self.regression[
                "wine_quality"
            ]
        )

        self.assertEqual(
            resolve_dataset_params(
                self.regression,
                "wine_quality",
            ),
            regression_wine_expected,
        )

        self.assertEqual(
            resolve_dataset_params(
                self.regression,
                "dataset_without_override",
            ),
            self.regression[
                "__default__"
            ],
        )


if __name__ == "__main__":
    unittest.main()
