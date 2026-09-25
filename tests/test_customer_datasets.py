from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.customer_datasets import make_customer_dataset, parse_uploaded_csv
from src.persona_generation import build_persona_model_input


FIXTURE = Path(__file__).parent / "fixtures" / "custom_customers.csv"


class CustomerDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = parse_uploaded_csv(FIXTURE.read_text(encoding="utf-8"))
        self.dataset = make_customer_dataset("fixture", FIXTURE.name, self.df)

    def test_infers_descriptive_fields_and_excludes_identifiers(self) -> None:
        self.assertEqual(
            self.dataset.clusterable_fields,
            ("Role", "Vertical", "Region", "Product Notes"),
        )
        self.assertNotIn("Contact Name", self.dataset.clusterable_fields)
        self.assertNotIn("ARR", self.dataset.clusterable_fields)
        self.assertNotIn("Website URL", self.dataset.clusterable_fields)

    def test_chooses_high_signal_defaults(self) -> None:
        self.assertEqual(
            self.dataset.default_fields,
            ("Role", "Vertical", "Product Notes"),
        )
        self.assertEqual(self.dataset.display_fields, ("Contact Name", "Role", "Vertical"))

    def test_persona_input_includes_every_selected_field(self) -> None:
        fields = list(self.dataset.default_fields)
        memberships = np.array(
            [
                [0.9, 0.1], [0.8, 0.2], [0.2, 0.8],
                [0.1, 0.9], [0.6, 0.4], [0.4, 0.6],
            ]
        )
        candidates = build_persona_model_input(self.df, fields, {"2": memberships})
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertEqual(candidate["selected_fields"], fields)
            self.assertEqual(set(candidate["field_evidence"]), set(fields))
            self.assertTrue(candidate["representative_rows"])


if __name__ == "__main__":
    unittest.main()
