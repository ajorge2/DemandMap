from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.news_query import (
    INITIAL_MAX_OUTPUT_TOKENS,
    NEWSAPI_QUERY_TARGET,
    RETRY_MAX_OUTPUT_TOKENS,
    _query_syntax_error,
    generate_news_query,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _completed_response() -> dict[str, object]:
    query = {
        "q": "fintech AND regulation",
        "market_terms": ["fintech"],
        "event_terms": ["regulation"],
        "functional_terms": ["data"],
        "reasoning": "Surfaces regulatory changes affecting data operations.",
    }
    return {
        "status": "completed",
        "model": "gpt-5-mini-test",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(query)}],
            }
        ],
    }


class NewsQueryTests(unittest.TestCase):
    def test_detects_unbalanced_query_syntax(self) -> None:
        self.assertEqual(
            _query_syntax_error('(fintech OR "payments") AND (breach OR outage'),
            "query has an unmatched opening parenthesis",
        )
        self.assertEqual(
            _query_syntax_error('(fintech) AND "data breach'),
            "query has an unmatched double quote",
        )
        self.assertIsNone(_query_syntax_error('(fintech) AND ("data breach" OR outage)'))

    @patch("src.news_query.urllib.request.urlopen")
    def test_uses_larger_initial_output_budget(self, urlopen: object) -> None:
        urlopen.return_value = _FakeResponse(_completed_response())

        query, model = generate_news_query("test-key", ["A fintech data team"])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["max_output_tokens"], INITIAL_MAX_OUTPUT_TOKENS)
        self.assertEqual(query["q"], "fintech AND regulation")
        self.assertEqual(model, "gpt-5-mini-test")

    @patch("src.news_query.urllib.request.urlopen")
    def test_retries_once_after_output_token_exhaustion(self, urlopen: object) -> None:
        urlopen.side_effect = [
            _FakeResponse(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"output_tokens": INITIAL_MAX_OUTPUT_TOKENS},
                }
            ),
            _FakeResponse(_completed_response()),
        ]

        query, _ = generate_news_query("test-key", ["A fintech data team"])

        budgets = [
            json.loads(call.args[0].data.decode("utf-8"))["max_output_tokens"]
            for call in urlopen.call_args_list
        ]
        self.assertEqual(budgets, [INITIAL_MAX_OUTPUT_TOKENS, RETRY_MAX_OUTPUT_TOKENS])
        self.assertEqual(query["q"], "fintech AND regulation")

    @patch("src.news_query.urllib.request.urlopen")
    def test_reports_non_token_incomplete_reason_without_retry(self, urlopen: object) -> None:
        urlopen.return_value = _FakeResponse(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "usage": {"output_tokens": 321},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            r"reason: content_filter; output tokens used: 321; token budget: 6000",
        ):
            generate_news_query("test-key", ["A fintech data team"])

        self.assertEqual(urlopen.call_count, 1)

    @patch("src.news_query.urllib.request.urlopen")
    def test_reports_retry_exhaustion_details(self, urlopen: object) -> None:
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"output_tokens": RETRY_MAX_OUTPUT_TOKENS},
        }
        urlopen.side_effect = [_FakeResponse(incomplete), _FakeResponse(incomplete)]

        with self.assertRaisesRegex(
            ValueError,
            r"reason: max_output_tokens; output tokens used: 12000; token budget: 12000",
        ):
            generate_news_query("test-key", ["A fintech data team"])

        self.assertEqual(urlopen.call_count, 2)

    @patch("src.news_query.urllib.request.urlopen")
    def test_rebuilds_a_truncated_query_from_structured_terms(self, urlopen: object) -> None:
        response = _completed_response()
        output_text = response["output"][0]["content"][0]
        query = json.loads(output_text["text"])
        query["q"] = '(fintech OR payments) AND (regulation OR "data breach"'
        query["market_terms"] = ["fintech", "sports & entertainment"]
        query["event_terms"] = ["regulation", "data breach"]
        query["functional_terms"] = ["data operations", "security"]
        output_text["text"] = json.dumps(query)
        urlopen.return_value = _FakeResponse(response)

        repaired, _ = generate_news_query("test-key", ["A fintech data team"])

        self.assertIsNone(_query_syntax_error(str(repaired["q"])))
        self.assertLessEqual(len(str(repaired["q"])), NEWSAPI_QUERY_TARGET)
        self.assertIn('"sports and entertainment"', str(repaired["q"]))


if __name__ == "__main__":
    unittest.main()
