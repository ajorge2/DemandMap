from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from src.persona_generation import (
    MAX_EVIDENCE_VALUE_CHARACTERS,
    build_persona_model_input,
    validate_and_resolve_persona_output,
)


class PersonaGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = ["Role", "Company", "Sparse"]
        self.df = pd.DataFrame(
            {
                "Role": ["Head of Data", "Analytics Director", "Security Lead"],
                "Company": ["Acme <script>alert(1)</script>", "Beta", "Gamma"],
                "Sparse": [None, "", np.nan],
            }
        )
        memberships = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
        self.candidates = build_persona_model_input(
            self.df, self.fields, {"2": memberships}
        )
        self.output = [self._valid_candidate(candidate) for candidate in self.candidates]

    def _field_ids(self, candidate: dict[str, object], field: str) -> list[str]:
        return [
            str(item["evidence_id"])
            for item in candidate["evidence_items"]
            if item["field"] == field
        ]

    def _valid_candidate(self, candidate: dict[str, object]) -> dict[str, object]:
        populated_ids = [
            str(item["evidence_id"]) for item in candidate["evidence_items"]
        ]
        summaries = []
        for field in self.fields:
            ids = self._field_ids(candidate, field)
            summaries.append(
                {
                    "field": field,
                    "summary": (
                        f"Observed representative {field.lower()} evidence."
                        if ids else "No representative evidence was available."
                    ),
                    "evidence_status": "supported" if ids else "no_evidence",
                    "evidence_ids": ids[:1],
                }
            )
        return {
            "candidate_id": candidate["candidate_id"],
            "name": "Data and security leaders",
            "description": "Leaders represented by the supplied customer records.",
            "description_evidence_ids": populated_ids[:2],
            "field_summaries": summaries,
        }

    def test_evidence_ids_are_deterministic_and_candidate_scoped(self) -> None:
        rebuilt = build_persona_model_input(
            self.df,
            self.fields,
            {"2": np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])},
        )
        first_ids = [item["evidence_id"] for item in self.candidates[0]["evidence_items"]]
        rebuilt_ids = [item["evidence_id"] for item in rebuilt[0]["evidence_items"]]
        second_ids = [item["evidence_id"] for item in self.candidates[1]["evidence_items"]]
        self.assertEqual(first_ids, rebuilt_ids)
        self.assertTrue(set(first_ids).isdisjoint(second_ids))
        for item in self.candidates[0]["evidence_items"]:
            self.assertEqual(item["candidate_id"], "K2-C0")
            self.assertTrue(str(item["row_id"]).startswith("source_row_"))
            self.assertLessEqual(len(str(item["value"])), MAX_EVIDENCE_VALUE_CHARACTERS)

    def test_valid_references_resolve_server_owned_exact_values(self) -> None:
        resolved = validate_and_resolve_persona_output(
            self.output, self.candidates, self.fields
        )
        candidate = resolved["K2-C0"]
        self.assertEqual(len(candidate["field_summaries"]), len(self.fields))
        source = candidate["source_evidence"]
        self.assertTrue(any("<script>" in item["value"] for item in source))
        sparse = next(
            item for item in candidate["field_summaries"] if item["field"] == "Sparse"
        )
        self.assertEqual(sparse["evidenceStatus"], "no_evidence")
        self.assertEqual(sparse["evidence"], [])

    def test_rejects_unknown_reference(self) -> None:
        output = copy.deepcopy(self.output)
        output[0]["description_evidence_ids"] = ["ev_unknown"]
        with self.assertRaisesRegex(ValueError, "unknown evidence ID"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)

    def test_rejects_unknown_or_duplicate_candidate_ids(self) -> None:
        unknown = copy.deepcopy(self.output)
        unknown[0]["candidate_id"] = "K2-C99"
        with self.assertRaisesRegex(ValueError, "unknown or duplicate candidate ID"):
            validate_and_resolve_persona_output(unknown, self.candidates, self.fields)
        duplicate = copy.deepcopy(self.output)
        duplicate[1]["candidate_id"] = duplicate[0]["candidate_id"]
        with self.assertRaisesRegex(ValueError, "unknown or duplicate candidate ID"):
            validate_and_resolve_persona_output(duplicate, self.candidates, self.fields)

    def test_rejects_missing_or_duplicate_selected_field_summary(self) -> None:
        missing = copy.deepcopy(self.output)
        missing[0]["field_summaries"].pop()
        with self.assertRaisesRegex(ValueError, "every selected CSV field"):
            validate_and_resolve_persona_output(missing, self.candidates, self.fields)
        duplicate = copy.deepcopy(self.output)
        duplicate[0]["field_summaries"][2] = copy.deepcopy(
            duplicate[0]["field_summaries"][0]
        )
        with self.assertRaisesRegex(ValueError, "every selected CSV field"):
            validate_and_resolve_persona_output(duplicate, self.candidates, self.fields)

    def test_rejects_cross_candidate_reference(self) -> None:
        output = copy.deepcopy(self.output)
        foreign = self.candidates[1]["evidence_items"][0]["evidence_id"]
        output[0]["description_evidence_ids"] = [foreign]
        with self.assertRaisesRegex(ValueError, "cross-referenced"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)

    def test_rejects_wrong_field_reference(self) -> None:
        output = copy.deepcopy(self.output)
        company_id = self._field_ids(self.candidates[0], "Company")[0]
        role_summary = next(
            item for item in output[0]["field_summaries"] if item["field"] == "Role"
        )
        role_summary["evidence_ids"] = [company_id]
        with self.assertRaisesRegex(ValueError, "wrong-field evidence"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)

    def test_rejects_duplicate_reference(self) -> None:
        output = copy.deepcopy(self.output)
        evidence_id = output[0]["description_evidence_ids"][0]
        output[0]["description_evidence_ids"] = [evidence_id, evidence_id]
        with self.assertRaisesRegex(ValueError, "duplicate evidence references"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)

    def test_rejects_populated_summary_without_reference(self) -> None:
        output = copy.deepcopy(self.output)
        role_summary = next(
            item for item in output[0]["field_summaries"] if item["field"] == "Role"
        )
        role_summary["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "populated field Role has no valid evidence"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)

    def test_rejects_fabricated_reference_for_sparse_field(self) -> None:
        output = copy.deepcopy(self.output)
        sparse_summary = next(
            item for item in output[0]["field_summaries"] if item["field"] == "Sparse"
        )
        sparse_summary["evidence_status"] = "supported"
        sparse_summary["evidence_ids"] = self._field_ids(self.candidates[0], "Role")[:1]
        with self.assertRaisesRegex(ValueError, "wrong-field evidence"):
            validate_and_resolve_persona_output(output, self.candidates, self.fields)


if __name__ == "__main__":
    unittest.main()
