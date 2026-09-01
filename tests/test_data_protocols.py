from __future__ import annotations

import json
import unittest

from pathlib import Path

import numpy as np

import opns_pack.custom_gen_pairs as custom_gen_pairs


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


if __name__ == "__main__":
    unittest.main()
