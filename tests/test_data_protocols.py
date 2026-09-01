from __future__ import annotations

import json
import unittest

from pathlib import Path

import numpy as np
import pandas as pd

import opns_pack.custom_gen_pairs as custom_gen_pairs
import opns_boost.data as data_module


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "all_pair_repeat_frozen_probe.json"
)


def normalize(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return normalize(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        tuple,
    ):
        return [
            normalize(item)
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            normalize(item)
            for item in value
        ]

    return value


class TestDataProtocols(unittest.TestCase):
    def test_all_pair_repeat_matches_frozen_probe(self) -> None:
        fixture = json.loads(
            FIXTURE_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            hasattr(
                custom_gen_pairs,
                "all_pair_repeat",
            )
        )

        for case in fixture[
            "cases"
        ]:
            with self.subTest(
                case=case[
                    "name"
                ]
            ):
                actual = normalize(
                    custom_gen_pairs.all_pair_repeat(
                        list(
                            case[
                                "features"
                            ]
                        )
                    )
                )

                self.assertEqual(
                    actual,
                    case[
                        "result"
                    ],
                )


class TestMigratedDataProtocols(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_root = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "data_protocols"
        )

        cls.expected = json.loads(
            (
                cls.protocol_root
                / "expected.json"
            ).read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _frame_values(frame):
        return [
            [
                None
                if np.isnan(value)
                else float(value)
                for value
                in row
            ]
            for row
            in frame.to_numpy(
                dtype=float
            )
        ]

    @staticmethod
    def _opns_values(value):
        return {
            "shape": list(
                value.shape
            ),
            "left": (
                np.asarray(
                    value.left_matrix,
                    dtype=float,
                )
            ),
            "right": (
                np.asarray(
                    value.right_matrix,
                    dtype=float,
                )
            ),
        }

    def test_feature_construction_matches_frozen_fixture(
        self,
    ) -> None:
        synthetic = pd.DataFrame(
            {
                "a": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                "b": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                "c": [
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                ],
            }
        )

        for case in self.expected[
            "feature_cases"
        ]:
            with self.subTest(
                pairing=case[
                    "pairing"
                ],
                scale=case[
                    "scale"
                ],
            ):
                matrix, names = (
                    data_module.make_opns_features(
                        synthetic,
                        mode=case[
                            "pairing"
                        ],
                        scale_features=case[
                            "scale"
                        ],
                    )
                )

                self.assertEqual(
                    [
                        str(name)
                        for name
                        in names
                    ],
                    case[
                        "pair_features"
                    ],
                )

                actual = (
                    self._opns_values(
                        matrix
                    )
                )

                self.assertEqual(
                    actual[
                        "shape"
                    ],
                    case[
                        "opns"
                    ][
                        "shape"
                    ],
                )

                np.testing.assert_allclose(
                    actual[
                        "left"
                    ],
                    np.asarray(
                        case[
                            "opns"
                        ][
                            "left"
                        ],
                        dtype=float,
                    ),
                    rtol=0.0,
                    atol=1.0e-12,
                )

                np.testing.assert_allclose(
                    actual[
                        "right"
                    ],
                    np.asarray(
                        case[
                            "opns"
                        ][
                            "right"
                        ],
                        dtype=float,
                    ),
                    rtol=0.0,
                    atol=1.0e-12,
                )

    def test_wine_loader_preserves_raw_missingness(
        self,
    ) -> None:
        expected = self.expected[
            "wine"
        ]

        dataset = (
            data_module.load_regression_dataset(
                "wine_quality",
                data_root=self.protocol_root,
                defer_imputation=True,
            )
        )

        self.assertEqual(
            list(
                dataset.feature_names
            ),
            expected[
                "feature_names"
            ],
        )

        self.assertEqual(
            int(
                dataset.X
                .isna()
                .sum()
                .sum()
            ),
            expected[
                "raw_missing_total"
            ],
        )

        self.assertEqual(
            self._frame_values(
                dataset.X
            ),
            expected[
                "raw_X"
            ],
        )

    def test_wine_fold_local_imputation_matches_frozen(
        self,
    ) -> None:
        expected = self.expected[
            "wine"
        ][
            "prepared"
        ]

        dataset = (
            data_module.load_regression_dataset(
                "wine_quality",
                data_root=self.protocol_root,
                defer_imputation=True,
            )
        )

        prepared = (
            data_module.prepare_regression_fold(
                dataset,
                np.asarray(
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                    dtype=int,
                ),
                np.asarray(
                    [
                        4,
                        5,
                    ],
                    dtype=int,
                ),
                need_opns=True,
                fold_local_imputation=True,
                pairing="combinations",
                scale_features="none",
            )
        )

        self.assertEqual(
            self._frame_values(
                prepared.X_train_frame
            ),
            expected[
                "X_train_frame"
            ],
        )

        self.assertEqual(
            self._frame_values(
                prepared.X_test_frame
            ),
            expected[
                "X_test_frame"
            ],
        )

        # Leakage guard:
        # training fixed-acidity values are
        # [7, 6, 8, 9], therefore train median = 7.5.
        # The missing test-fold value must use that median,
        # not information from the held-out row containing 100.
        self.assertAlmostEqual(
            float(
                prepared.X_test_frame.iloc[
                    0,
                    1,
                ]
            ),
            7.5,
            places=12,
        )

        actual_train = (
            self._opns_values(
                prepared.X_train_opns
            )
        )

        actual_test = (
            self._opns_values(
                prepared.X_test_opns
            )
        )

        np.testing.assert_allclose(
            actual_train[
                "left"
            ],
            np.asarray(
                expected[
                    "X_train_opns"
                ][
                    "left"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            actual_train[
                "right"
            ],
            np.asarray(
                expected[
                    "X_train_opns"
                ][
                    "right"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            actual_test[
                "left"
            ],
            np.asarray(
                expected[
                    "X_test_opns"
                ][
                    "left"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            actual_test[
                "right"
            ],
            np.asarray(
                expected[
                    "X_test_opns"
                ][
                    "right"
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_corrected_car_persons_encoding_matches_frozen(
        self,
    ) -> None:
        expected = self.expected[
            "car"
        ]

        dataset = (
            data_module.load_classification_dataset(
                "car",
                data_root=self.protocol_root,
            )
        )

        self.assertEqual(
            list(
                dataset.feature_names
            ),
            expected[
                "feature_names"
            ],
        )

        persons = [
            float(value)
            for value
            in dataset.X[
                "persons"
            ]
        ]

        self.assertEqual(
            persons,
            expected[
                "persons"
            ],
        )

        self.assertEqual(
            persons,
            [
                0.0,
                1.0,
                2.0,
            ],
        )


if __name__ == "__main__":
    unittest.main()
