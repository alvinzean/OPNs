from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest

from pathlib import Path

from research.hybridboost.scripts import (
    reproduce_overall,
)
from research.hybridboost.scripts import (
    _classification_baselines,
)
from research.hybridboost.scripts import (
    _regression_baselines,
)


ROOT = (
    Path(__file__).resolve().parents[1]
)

FIXTURE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "overall_reproduction_contract.json"
)


def function_ast_hashes(
    path: Path,
):
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
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


class TestReproduceOverall(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            FIXTURE_PATH.read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_helper_ast_contract(
        self,
    ) -> None:
        classification_path = Path(
            _classification_baselines.__file__
        )

        regression_path = Path(
            _regression_baselines.__file__
        )

        actual_class = (
            function_ast_hashes(
                classification_path
            )
        )

        actual_reg = (
            function_ast_hashes(
                regression_path
            )
        )

        for name, expected in (
            self.fixture[
                "classification_helper_ast_sha256"
            ].items()
        ):
            self.assertEqual(
                actual_class[
                    name
                ],
                expected,
            )

        for name, expected in (
            self.fixture[
                "regression_helper_ast_sha256"
            ].items()
        ):
            self.assertEqual(
                actual_reg[
                    name
                ],
                expected,
            )

    def test_default_protocol_matches_formal_contract(
        self,
    ) -> None:
        args = (
            reproduce_overall.parse_args(
                [
                    "--dry-run",
                ]
            )
        )

        protocol = (
            reproduce_overall.build_protocol(
                args
            )
        )

        expected = self.fixture[
            "formal_protocol"
        ]

        self.assertEqual(
            args.random_state,
            expected[
                "random_state"
            ],
        )

        self.assertEqual(
            args.n_splits,
            expected[
                "n_splits"
            ],
        )

        self.assertEqual(
            args.n_repeats,
            expected[
                "n_repeats"
            ],
        )

        self.assertEqual(
            reproduce_overall.SUMMARY_DDOF,
            expected[
                "summary_ddof"
            ],
        )

        self.assertEqual(
            args.classification_datasets,
            expected[
                "classification_datasets"
            ],
        )

        self.assertEqual(
            args.regression_datasets,
            expected[
                "regression_datasets"
            ],
        )

        self.assertEqual(
            args.models,
            expected[
                "models"
            ],
        )

        self.assertEqual(
            protocol[
                "regression"
            ][
                "wine_quality_imputation"
            ],
            (
                "training-fold median only"
            ),
        )

    def test_wine_fold_local_policy(
        self,
    ) -> None:
        self.assertTrue(
            reproduce_overall
            .uses_fold_local_imputation(
                "wine_quality"
            )
        )

        self.assertFalse(
            reproduce_overall
            .uses_fold_local_imputation(
                "concrete"
            )
        )

    def test_final_histgb_baseline_resolution(
        self,
    ) -> None:
        class_config = (
            reproduce_overall.load_json(
                reproduce_overall
                .DEFAULT_CLASSIFICATION_BASELINES
            )
        )

        class_overrides = (
            _classification_baselines
            .merge_nested_params(
                class_config,
                "car",
                "histgb",
            )
        )

        classifier = (
            _classification_baselines
            .build_model(
                "histgb",
                n_classes=4,
                n_estimators=60,
                learning_rate=0.1,
                max_depth=3,
                l2_leaf_reg=1.0,
                random_state=161803,
                n_jobs=1,
                overrides=class_overrides,
            )
        )

        self.assertEqual(
            classifier.max_iter,
            160,
        )

        self.assertEqual(
            classifier.max_depth,
            4,
        )

        self.assertEqual(
            classifier.max_leaf_nodes,
            16,
        )

        reg_config = (
            reproduce_overall.load_json(
                reproduce_overall
                .DEFAULT_REGRESSION_BASELINES
            )
        )

        opns_config = (
            reproduce_overall.load_json(
                reproduce_overall
                .DEFAULT_REGRESSION_CONFIG
            )
        )

        common = (
            _regression_baselines
            .map_common_budget(
                reproduce_overall
                .resolve_dataset_params(
                    opns_config,
                    "wine_quality",
                )
            )
        )

        reg_overrides = (
            _regression_baselines
            .merge_nested_params(
                reg_config,
                "wine_quality",
                "histgb",
            )
        )

        regressor = (
            _regression_baselines
            .build_external_model(
                "histgb",
                common,
                161803,
                1,
                reg_overrides,
            )
        )

        for key, value in (
            reg_overrides.items()
        ):
            self.assertEqual(
                getattr(
                    regressor,
                    key
                ),
                value,
            )

    def test_dry_run_writes_reproducibility_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(
                tmp
            )

            status = (
                reproduce_overall.main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(
                            output
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
                    output
                    / "protocol.json"
                ).is_file()
            )

            self.assertTrue(
                (
                    output
                    / "environment.json"
                ).is_file()
            )

            for name in (
                "classification_final.json",
                "regression_final.json",
                "classification_baselines_final.json",
                "regression_baselines_final.json",
            ):
                self.assertTrue(
                    (
                        output
                        / name
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
