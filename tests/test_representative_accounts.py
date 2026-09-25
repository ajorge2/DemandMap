from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.representative_accounts import MISSING_VALUE, build_representative_accounts


class RepresentativeAccountsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "Account": ["Alpha", "Beta", "Gamma", "Delta"],
                "Role": ["VP Data", "VP Data", "Head of AI", "COO"],
                "Market": ["Fintech", None, "Payments", "Logistics"],
            }
        )
        self.memberships = np.array(
            [
                [0.82, 0.91],
                [0.82, 0.10],
                [0.95, 0.91],
                [0.20, 0.30],
            ]
        )

    def test_ranks_each_persona_by_independent_affinity(self) -> None:
        result = build_representative_accounts(
            self.dataframe,
            self.memberships,
            ["Role", "Market"],
            ["Account", "Role"],
            limit=3,
        )
        self.assertEqual(
            [example["rowId"] for example in result[0]],
            ["source_row_2", "source_row_0", "source_row_1"],
        )
        self.assertEqual(
            [example["rowId"] for example in result[1]],
            ["source_row_0", "source_row_2", "source_row_3"],
        )
        self.assertEqual(result[0][0]["score"], 0.95)

    def test_breaks_ties_by_source_row_and_allows_overlap(self) -> None:
        result = build_representative_accounts(
            self.dataframe,
            self.memberships,
            ["Role"],
            ["Account"],
            limit=2,
        )
        self.assertEqual(
            [example["rowId"] for example in result[1]],
            ["source_row_0", "source_row_2"],
        )
        self.assertIn("source_row_2", {example["rowId"] for example in result[0]})
        self.assertIn("source_row_2", {example["rowId"] for example in result[1]})

    def test_exposes_display_values_all_fields_and_explicit_missing_values(self) -> None:
        result = build_representative_accounts(
            self.dataframe,
            self.memberships,
            ["Role", "Market"],
            ["Account", "Role"],
            limit=3,
        )
        sparse_example = next(
            example for example in result[0] if example["rowId"] == "source_row_1"
        )
        self.assertEqual(sparse_example["displayValues"]["Account"], "Beta")
        self.assertEqual(set(sparse_example["fieldValues"]), {"Role", "Market"})
        self.assertEqual(sparse_example["fieldValues"]["Market"], MISSING_VALUE)

    def test_limits_examples_to_dataset_size_and_truncates_long_values(self) -> None:
        dataframe = self.dataframe.iloc[:2].copy()
        dataframe.loc[0, "Role"] = "A" * 30
        result = build_representative_accounts(
            dataframe,
            self.memberships[:2],
            ["Role"],
            ["Account"],
            limit=10,
            max_value_characters=12,
        )
        self.assertEqual([len(examples) for examples in result], [2, 2])
        alpha = next(example for example in result[0] if example["rowId"] == "source_row_0")
        self.assertEqual(alpha["fieldValues"]["Role"], "A" * 11 + "…")


if __name__ == "__main__":
    unittest.main()
