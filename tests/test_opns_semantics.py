from __future__ import annotations

import unittest

import opns_pack.custom_gen_pairs as custom_gen_pairs
import opns_pack.opns_np as opns_np
import opns_module.linear_model as linear_model

from opns_pack.opns import OPNs
from opns_pack.opns_matrix import OPNsMatrix
from opns_module.preprocessing import OPNsStandardScaler


class TestOPNsSemantics(unittest.TestCase):
    """Compatibility guard for the shared OPNs foundation.

    These tests record behavior and public names already exposed by the
    repository before OPNs-HybridBoost is integrated.
    """

    def test_reverse_pairs_remain_distinct_and_ordered(self) -> None:
        forward = OPNs(1.0, 2.0)
        reverse = OPNs(2.0, 1.0)

        self.assertNotEqual(forward, reverse)
        self.assertTrue(forward < reverse)
        self.assertTrue(reverse > forward)

    def test_basic_addition_and_multiplication(self) -> None:
        left = OPNs(1.0, 2.0)
        right = OPNs(3.0, 4.0)

        added = left + right
        multiplied = left * right

        self.assertEqual(added, OPNs(4.0, 6.0))
        self.assertEqual(multiplied, OPNs(-10.0, -11.0))

    def test_matrix_roundtrip_preserves_pair_order(self) -> None:
        forward = OPNs(1.0, 2.0)
        reverse = OPNs(2.0, 1.0)

        matrix = opns_np.array(
            [
                forward,
                reverse,
            ]
        )

        self.assertIsInstance(matrix, OPNsMatrix)

        flattened = matrix.flatten()

        self.assertEqual(flattened[0], forward)
        self.assertEqual(flattened[1], reverse)

    def test_existing_opns_numpy_compatibility_names(self) -> None:
        required_names = (
            "array",
            "dot",
            "mean",
            "std",
            "zeros",
            "ones",
            "opns_to_1_num",
            "opns_to_2_num",
        )

        for name in required_names:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(opns_np, name),
                    msg=f"Missing public opns_np name: {name}",
                )
                self.assertTrue(
                    callable(getattr(opns_np, name)),
                    msg=f"Public opns_np name is not callable: {name}",
                )

    def test_existing_linear_model_classes_remain_available(self) -> None:
        required_classes = (
            "LinearRegression",
            "LinearRegressionGradientDescent",
            "Lasso",
        )

        for name in required_classes:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(linear_model, name),
                    msg=f"Missing public linear-model class: {name}",
                )
                self.assertTrue(
                    isinstance(getattr(linear_model, name), type),
                    msg=f"Public linear-model name is not a class: {name}",
                )

    def test_existing_preprocessing_class_remains_available(self) -> None:
        self.assertTrue(
            isinstance(OPNsStandardScaler, type)
        )

        scaler = OPNsStandardScaler()

        self.assertTrue(
            callable(getattr(scaler, "fit", None))
        )
        self.assertTrue(
            callable(getattr(scaler, "transform", None))
        )
        self.assertTrue(
            callable(getattr(scaler, "fit_transform", None))
        )

    def test_existing_pair_generation_contract(self) -> None:
        features = [
            "x",
            "y",
            "z",
        ]

        self.assertEqual(
            custom_gen_pairs.all_pair(features),
            [
                "x",
                "y",
                "x",
                "z",
                "y",
                "z",
            ],
        )

        self.assertEqual(
            custom_gen_pairs.linear_pair(
                features
            ),
            [
                "x",
                "zero",
                "y",
                "zero",
                "z",
                "zero",
            ],
        )


    def test_vstack_accepts_one_dimensional_vectors(self) -> None:
        first = opns_np.array(
            [
                OPNs(1.0, 2.0),
                OPNs(3.0, 4.0),
            ]
        )

        second = opns_np.array(
            [
                OPNs(5.0, 6.0),
                OPNs(7.0, 8.0),
            ]
        )

        self.assertEqual(
            first.shape,
            (2,),
        )

        self.assertEqual(
            second.shape,
            (2,),
        )

        stacked = opns_np.vstack(
            first,
            second,
        )

        self.assertEqual(
            stacked.shape,
            (
                2,
                2,
            ),
        )

        flattened = stacked.flatten()

        expected = [
            OPNs(1.0, 2.0),
            OPNs(3.0, 4.0),
            OPNs(5.0, 6.0),
            OPNs(7.0, 8.0),
        ]

        self.assertEqual(
            len(flattened),
            len(expected),
        )

        for actual, target in zip(
            flattened,
            expected,
        ):
            self.assertEqual(
                actual,
                target,
            )


if __name__ == "__main__":
    unittest.main()
