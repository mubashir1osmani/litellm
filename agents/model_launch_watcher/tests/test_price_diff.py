"""Price-drift detection: the check that catches a price cut."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from model_launch_watcher.catalog import Catalog
from model_launch_watcher.domain import Confidence, Provenance, Sourced, TokenPricing
from model_launch_watcher.memory import CorrectionKind, Memory, correction
from model_launch_watcher.price_diff import (
    before_after_table,
    diff_entry,
    price_drift_candidates,
    resolve_catalog_key,
    slugify,
)

SOURCE = "https://docs.claude.com/pricing"


def provenance() -> Provenance:
    return Provenance(source_url=SOURCE, retrieved_at=datetime(2026, 8, 20, tzinfo=UTC), confidence=Confidence.PRIMARY_DOC)


def pricing(input_per_token: float, output_per_token: float) -> TokenPricing:
    return TokenPricing(
        input_cost_per_token=Sourced(input_per_token, provenance()),
        output_cost_per_token=Sourced(output_per_token, provenance()),
    )


def catalog() -> Catalog:
    return Catalog(
        entries=MappingProxyType(
            {
                "claude-opus-5": MappingProxyType(
                    {"input_cost_per_token": 5e-06, "output_cost_per_token": 2.5e-05, "mode": "chat"}
                )
            }
        )
    )


def memory(tmp_path: Path) -> Memory:
    return Memory.load(tmp_path / "corrections.jsonl")


def test_detects_a_price_cut(tmp_path: Path) -> None:
    candidates, _ = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5": pricing(3e-06, 2.5e-05)}, memory(tmp_path)
    )
    assert len(candidates) == 1
    delta = candidates[0].deltas[0]
    assert delta.field == "input_cost_per_token"
    assert delta.direction == "cut"
    assert delta.percent_change == pytest.approx(-40.0)


def test_detects_a_price_increase(tmp_path: Path) -> None:
    candidates, _ = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5": pricing(5e-06, 5e-05)}, memory(tmp_path)
    )
    assert candidates[0].deltas[0].direction == "increase"


def test_reports_nothing_when_prices_agree(tmp_path: Path) -> None:
    candidates, failures = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5": pricing(5e-06, 2.5e-05)}, memory(tmp_path)
    )
    assert candidates == ()
    assert failures == ()


def test_unmappable_name_is_reported_not_guessed(tmp_path: Path) -> None:
    """Binding a price to the wrong model is worse than reporting nothing."""
    candidates, failures = price_drift_candidates(
        "anthropic", catalog(), {"Claude Haiku 3.5 (retired)": pricing(1e-06, 5e-06)}, memory(tmp_path)
    )
    assert candidates == ()
    assert [f.reason for f in failures] == ["needs_mapping"]


def test_memory_binds_a_name_the_agent_could_not_infer(tmp_path: Path) -> None:
    taught = memory(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Claude Opus 5 (latest)", "claude-opus-5", "alias", "sre")
    )
    candidates, failures = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5 (latest)": pricing(3e-06, 2.5e-05)}, taught
    )
    assert failures == ()
    assert candidates[0].catalog_key == "claude-opus-5"


def test_memory_pointing_at_a_missing_key_is_flagged_as_stale(tmp_path: Path) -> None:
    taught = memory(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Ghost Model", "claude-does-not-exist", "typo", "sre")
    )
    _, failures = price_drift_candidates("anthropic", catalog(), {"Ghost Model": pricing(1e-06, 2e-06)}, taught)
    assert [f.reason for f in failures] == ["mapping_stale"]


def test_pinned_field_stops_recurring(tmp_path: Path) -> None:
    """A pin says the catalogued value is deliberate, so it must not be re-reported."""
    pinned = memory(tmp_path).record(
        correction(CorrectionKind.PIN_VALUE, "claude-opus-5", "input_cost_per_token", "5e-06", "negotiated", "sre")
    )
    deltas = diff_entry("claude-opus-5", catalog().entries["claude-opus-5"], pricing(3e-06, 2.5e-05), pinned)
    assert deltas == ()


def test_suppression_silences_the_finding(tmp_path: Path) -> None:
    silenced = memory(tmp_path).record(
        correction(CorrectionKind.SUPPRESS, "claude-opus-5", "price_drift", "", "tracked elsewhere", "sre")
    )
    candidates, _ = price_drift_candidates("anthropic", catalog(), {"Claude Opus 5": pricing(3e-06, 2.5e-05)}, silenced)
    assert candidates == ()


def test_floating_point_noise_is_not_a_price_change(tmp_path: Path) -> None:
    candidates, _ = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5": pricing(5e-06 + 1e-18, 2.5e-05)}, memory(tmp_path)
    )
    assert candidates == ()


def test_resolution_prefers_memory_over_inference(tmp_path: Path) -> None:
    taught = memory(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Claude Opus 5", "claude-opus-5", "explicit", "sre")
    )
    assert resolve_catalog_key("anthropic", "Claude Opus 5", catalog(), taught).via == "memory"


def test_slugify_matches_the_catalog_naming_convention() -> None:
    assert slugify("Claude Opus 4.8") == "claude-opus-4-8"
    assert slugify("GPT-5.1 mini") == "gpt-5-1-mini"


def test_before_after_table_shows_dollars_per_million_and_the_source(tmp_path: Path) -> None:
    candidates, _ = price_drift_candidates(
        "anthropic", catalog(), {"Claude Opus 5": pricing(3e-06, 2.5e-05)}, memory(tmp_path)
    )
    table = before_after_table(candidates[0])
    assert "| input_cost_per_token | 5.0000 | 3.0000 | -40.0% |" in table
    assert SOURCE in table
