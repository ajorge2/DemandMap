from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.embeddings import generate_combined_field_embeddings, generate_embeddings


def _vectors_for(_api_key: str, texts: list[str], _model: str) -> list[np.ndarray]:
    vectors = []
    for text in texts:
        lowered = text.casefold()
        if "data" in lowered or "fintech" in lowered:
            vectors.append(np.asarray([1.0, 0.0, 0.0]))
        elif "sales" in lowered:
            vectors.append(np.asarray([0.0, 1.0, 0.0]))
        else:
            vectors.append(np.asarray([0.0, 0.0, 1.0]))
    return vectors


class OpenAIEmbeddingTests(unittest.TestCase):
    @patch("src.embeddings._request_openai_embeddings", side_effect=_vectors_for)
    def test_caches_unique_nonblank_inputs(self, request_embeddings: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            first = generate_embeddings(
                ["Data platform", "Sales platform", "Data platform", ""],
                backend="openai",
                model_name="text-embedding-test",
                api_key="test-key",
                cache_dir=cache_dir,
            )
            second = generate_embeddings(
                ["Data platform", "Sales platform", "Data platform", ""],
                backend="openai",
                model_name="text-embedding-test",
                api_key="test-key",
                cache_dir=cache_dir,
            )

        self.assertEqual(request_embeddings.call_count, 1)
        self.assertEqual(first.metadata["api_inputs"], 2)
        self.assertEqual(second.metadata["api_inputs"], 0)
        self.assertEqual(second.metadata["cache_hits"], 2)
        self.assertEqual(first.vectors.shape, (4, 3))
        np.testing.assert_allclose(first.vectors[0], first.vectors[2])
        np.testing.assert_allclose(first.vectors[3], np.zeros(3))

    @patch("src.embeddings._request_openai_embeddings", side_effect=_vectors_for)
    def test_combines_only_populated_selected_fields(self, _request_embeddings: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = generate_combined_field_embeddings(
                {
                    "Role": ["Data leader", "Sales leader", ""],
                    "Industry": ["Media", "", "Fintech"],
                },
                api_key="test-key",
                model_name="text-embedding-test",
                cache_dir=Path(temporary),
            )

        expected_first = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(result.vectors[0], expected_first)
        np.testing.assert_allclose(result.vectors[1], np.asarray([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(result.vectors[2], np.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(result.backend, "openai")
        self.assertEqual(result.metadata["api_inputs"], 4)
        self.assertIn("role data leader", result.normalized_texts[0])


if __name__ == "__main__":
    unittest.main()
