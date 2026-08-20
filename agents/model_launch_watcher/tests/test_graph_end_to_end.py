"""The graph wired end to end, with a fake extractor standing in for the network.

The per-module tests all bypass the graph, so a drift found by price_diff could still be
lost between nodes without any of them failing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

from model_launch_watcher.catalog import Catalog
from model_launch_watcher.domain import SourceFailure
from model_launch_watcher.graph import Dependencies, WatchRequest, run_watch
from model_launch_watcher.memory import Memory
from model_launch_watcher.pricing import ExtractedRow, ExtractedTable, PricingDoc, html_to_text

PAGE = "Claude Opus 5 $3 / MTok input and $25 / MTok output."
URL = "https://docs.claude.com/pricing"


class FakeExtractor:
    """Returns a table whose quotes really are present in the fake page."""

    async def extract_table(self, doc: PricingDoc, conventions: Sequence[str]) -> ExtractedTable:
        return ExtractedTable(
            doc=doc,
            rows=(
                ExtractedRow(
                    published_name="Claude Opus 5",
                    input_per_million=3.0,
                    output_per_million=25.0,
                    input_quote="Claude Opus 5 $3 / MTok",
                    output_quote="$25 / MTok output",
                ),
            ),
        )


class OfflineAggregator:
    source_url = "https://openrouter.ai/api/v1/models"

    async def prices(self, client: object) -> SourceFailure:
        return SourceFailure(source="openrouter", reason="unreachable", detail="offline in tests")


def stale_catalog() -> Catalog:
    return Catalog(
        entries=MappingProxyType(
            {
                "claude-opus-5": MappingProxyType(
                    {"mode": "chat", "input_cost_per_token": 8e-06, "output_cost_per_token": 2.5e-05}
                )
            }
        )
    )


def local_fetcher(html: str):
    """Feeds the real fetch path a page body without a network round trip."""

    async def fetch(provider: str, url: str, client: object) -> PricingDoc | SourceFailure:
        text = html_to_text(html)
        if len(text) < 2_000:
            return SourceFailure(source=provider, reason="doc_unusable", detail=f"{url}: {len(text)} chars")
        return PricingDoc(provider=provider, url=url, text=text, retrieved_at=datetime(2026, 8, 20, tzinfo=UTC))

    return fetch


def dependencies(tmp_path: Path, html: str) -> Dependencies:
    return Dependencies(
        catalog=stale_catalog(),
        memory=Memory.load(tmp_path / "corrections.jsonl"),
        sources=(),
        extractor=FakeExtractor(),
        aggregator=OfflineAggregator(),
        pricing_pages=MappingProxyType({"anthropic": URL}),
        fetch_doc=local_fetcher(html),
    )


async def test_drift_reaches_the_patch_through_the_whole_graph(tmp_path: Path) -> None:
    """Guards the display_name link between price_diff and the graph's price lookup."""
    report = await run_watch(
        dependencies(tmp_path, f"<html><body>{PAGE} {'padding. ' * 400}</body></html>"),
        WatchRequest(providers=("anthropic",), kinds=("price_drift",)),
    )
    assert [c.candidate.catalog_key for c in report.candidates] == ["claude-opus-5"]
    assert report.candidates[0].is_proposable
    assert report.patch.updates["claude-opus-5"]["input_cost_per_token"] == 3e-06
    assert "output_cost_per_token" not in report.patch.updates["claude-opus-5"]


async def test_coverage_reports_the_denominator(tmp_path: Path) -> None:
    """A drift count without its denominator reads as a clean bill of health."""
    report = await run_watch(
        dependencies(tmp_path, f"<html><body>{PAGE} {'padding. ' * 400}</body></html>"),
        WatchRequest(providers=("anthropic",), kinds=("price_drift",)),
    )
    assert report.coverage.compared == 1
    assert report.coverage.token_billed_entries == 1


async def test_an_unusable_page_yields_no_findings_and_a_recorded_gap(tmp_path: Path) -> None:
    report = await run_watch(
        dependencies(tmp_path, "<html><body>nothing here</body></html>"),
        WatchRequest(providers=("anthropic",), kinds=("price_drift",)),
    )
    assert report.candidates == ()
    assert any(f.reason == "doc_unusable" for f in report.failures)
    assert report.coverage.compared == 0
