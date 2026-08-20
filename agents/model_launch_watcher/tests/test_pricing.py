"""Grounding is the guard that stops a hallucinated price reaching the cost map.

These tests exist to fail if that guard is ever loosened.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from model_launch_watcher.domain import Confidence
from model_launch_watcher.pricing import (
    ExtractedQuote,
    ExtractedRow,
    ExtractedTable,
    PricingDoc,
    ground,
    ground_all,
    ground_table,
    html_to_text,
    parse_table,
)

PAGE_TEXT = "Claude Opus 5 $5 / MTok input and $25 / MTok output. Prompt caching read $0.50 / MTok."


def doc(text: str = PAGE_TEXT) -> PricingDoc:
    return PricingDoc(
        provider="anthropic",
        url="https://docs.claude.com/pricing",
        text=text,
        retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def test_ground_converts_per_million_to_per_token() -> None:
    sourced = ground(doc(), "Claude Opus 5 $5 / MTok", 5.0)
    assert sourced is not None
    assert sourced.value == pytest.approx(5e-06)
    assert sourced.provenance.confidence is Confidence.PRIMARY_DOC
    assert sourced.provenance.source_url == "https://docs.claude.com/pricing"


def test_ground_rejects_a_quote_absent_from_the_document() -> None:
    """The whole point: a fabricated snippet must not become a price."""
    assert ground(doc(), "Claude Opus 5 $3 / MTok", 3.0) is None


def test_ground_rejects_a_real_quote_carrying_a_different_number() -> None:
    """A model may quote a genuine line but report a number that is not in it."""
    assert ground(doc(), "Claude Opus 5 $5 / MTok", 3.0) is None


def test_ground_tolerates_whitespace_differences_only() -> None:
    assert ground(doc(), "Claude   Opus 5   $5 / MTok", 5.0) is not None


def test_ground_rejects_missing_and_negative_inputs() -> None:
    assert ground(doc(), None, 5.0) is None
    assert ground(doc(), "Claude Opus 5 $5 / MTok", None) is None
    assert ground(doc(), "Claude Opus 5 $5 / MTok", -5.0) is None


def test_ground_all_drops_only_the_ungrounded_field() -> None:
    quote = ExtractedQuote(
        input_per_million=5.0,
        output_per_million=99.0,
        input_quote="Claude Opus 5 $5 / MTok",
        output_quote="$99 / MTok output",
    )
    pricing = ground_all(doc(), quote)
    assert pricing.input_cost_per_token is not None
    assert pricing.output_cost_per_token is None
    assert not pricing.is_complete


def test_ground_table_keeps_only_rows_that_are_fully_grounded() -> None:
    table = ExtractedTable(
        doc=doc(),
        rows=(
            ExtractedRow(
                published_name="Claude Opus 5",
                input_per_million=5.0,
                output_per_million=25.0,
                input_quote="Claude Opus 5 $5 / MTok",
                output_quote="$25 / MTok output",
            ),
            ExtractedRow(
                published_name="Invented Model",
                input_per_million=1.0,
                output_per_million=2.0,
                input_quote="Invented Model $1 / MTok",
                output_quote="$2 / MTok",
            ),
        ),
    )
    grounded = ground_table(table)
    assert set(grounded) == {"Claude Opus 5"}


def test_parse_table_skips_malformed_rows_but_keeps_good_ones() -> None:
    payload = """```json
    {"models": [
      {"published_name": "Good", "input_per_million": 1, "output_per_million": 2,
       "input_quote": "q1", "output_quote": "q2"},
      {"published_name": "Missing prices"}
    ]}
    ```"""
    table = parse_table(payload, doc())
    assert isinstance(table, ExtractedTable)
    assert [r.published_name for r in table.rows] == ["Good"]


def test_parse_table_reports_malformed_json_as_a_gap() -> None:
    failure = parse_table("not json at all", doc())
    assert failure.reason == "extractor_malformed"


def test_html_to_text_strips_script_bodies() -> None:
    text = html_to_text("<html><script>var price = 999;</script><p>$5 / MTok</p></html>")
    assert "999" not in text
    assert "$5 / MTok" in text
