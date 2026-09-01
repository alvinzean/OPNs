from __future__ import annotations

import ast
import hashlib
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
from research.hybridboost.ablations.tree_only import (
    OPNsTreeOnlyRegressor,
)
from research.hybridboost.scripts import (
    _warm_start_classification,
)
from research.hybridboost.scripts import (
    _warm_start_regression,
)


ROOT = (
    Path(__file__).resolve().parents[1]
)

FIXTURE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "warm_start_helpers_frozen_probe.json"
)

CONFIG_DIR = (
    ROOT
    / "research"
    / "hybridboost"
    / "configs"
)


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def function_ast_hashes(
    path: Path,
):
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(
            path
        ),
    )

    result = {}

    for node in tree.body:
        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        payload = ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )

        result[
            node.name
        ] = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

    return result


class TestWarmStartHelpers(
    unittest.TestCase,
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.fixture = load_json(
            FIXTURE_PATH
        )

    def test_frozen_helper_ast_contract(
        self,
    ) -> None:
        cases = (
            (
                "classification",
                Path(
                    _warm_start_classification
                    .__file__
                ),
            ),
            (
                "regression",
                Path(
                    _warm_start_regression
                    .__file__
                ),
            ),
        )

        for task, path in cases:
            actual = (
                function_ast_hashes(
                    path
                )
            )

            expected = (
                self.fixture[
                    task
                ][
                    "function_ast_sha256"
                ]
            )

            for name, digest in (
                expected.items()
            ):
                with self.subTest(
                    task=task,
                    function=name,
                ):
                    self.assertEqual(
                        actual[
                            name
                        ],
                        digest,
                    )

    def test_classification_mode_constructor_contract(
        self,
    ) -> None:
        config = load_json(
            CONFIG_DIR
            / "classification_final.json"
        )

        params = (
            resolve_dataset_params(
                config,
                "iris",
            )
        )

        full = (
            _warm_start_classification
            .build_model(
                "full_all",
                params,
                161803,
            )
        )

        tree_only = (
            _warm_start_classification
            .build_model(
                "tree_only_all",
                params,
                161803,
            )
        )

        self.assertIsInstance(
            full,
            OPNsHybridClassifier,
        )

        self.assertIsInstance(
            tree_only,
            OPNsHybridClassifier,
        )

        self.assertTrue(
            full.phase1_enabled
        )

        self.assertFalse(
            tree_only.phase1_enabled
        )

        self.assertEqual(
            full.tree_feature_mode,
            "all",
        )

        self.assertEqual(
            tree_only.tree_feature_mode,
            "all",
        )

        self.assertEqual(
            full.random_state,
            161803,
        )

        self.assertEqual(
            tree_only.random_state,
            161803,
        )

        self.assertIsNone(
            full.early_stopping_rounds
        )

        self.assertIsNone(
            tree_only.early_stopping_rounds
        )

    def test_regression_mode_constructor_contract(
        self,
    ) -> None:
        config = load_json(
            CONFIG_DIR
            / "regression_warm_start_final.json"
        )

        params = (
            resolve_dataset_params(
                config,
                "airfoil",
            )
        )

        full = (
            _warm_start_regression
            .build_model(
                "full_all",
                params,
                161803,
            )
        )

        tree_only = (
            _warm_start_regression
            .build_model(
                "tree_only_all",
                params,
                161803,
            )
        )

        self.assertIsInstance(
            full,
            OPNsHybridRegressor,
        )

        self.assertIsInstance(
            tree_only,
            OPNsTreeOnlyRegressor,
        )

        self.assertEqual(
            full.tree_feature_mode,
            "all",
        )

        self.assertEqual(
            tree_only.tree_feature_mode,
            "all",
        )

        self.assertEqual(
            full.prediction_backend,
            "object_batch",
        )

        self.assertEqual(
            tree_only.prediction_backend,
            "object_batch",
        )

        self.assertEqual(
            full.random_state,
            161803,
        )

        self.assertEqual(
            tree_only.random_state,
            161803,
        )


if __name__ == "__main__":
    unittest.main()
