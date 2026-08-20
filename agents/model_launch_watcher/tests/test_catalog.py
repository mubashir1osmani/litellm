"""Reconciliation must not cry wolf: most catalogued models are not billed per token."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from model_launch_watcher.catalog import Catalog, build_patch, catalog_keys_for
from model_launch_watcher.domain import (
    Confidence,
    Inventory,
    LiveModel,
    PricedCandidate,
    Provenance,
    Sourced,
    TokenPricing,
)


def provenance() -> Provenance:
    return Provenance(
        source_url="https://example.test/pricing",
        retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
        confidence=Confidence.PRIMARY_DOC,
    )


def inventory(*models: LiveModel) -> Inventory:
    return Inventory(provider=models[0].provider, models=models, provenance=provenance())


def catalog(entries: dict[str, dict[str, object]]) -> Catalog:
    return Catalog(entries=MappingProxyType({k: MappingProxyType(v) for k, v in entries.items()}))


def test_speech_model_priced_per_character_is_not_a_missing_price() -> None:
    """audio_speech bills per character, so an absent per-token cost is correct, not a gap."""
    found = catalog({"tts-1-hd": {"input_cost_per_character": 3e-05, "mode": "audio_speech"}}).reconcile(
        inventory(LiveModel(provider="openai", model_id="tts-1-hd"))
    )
    assert found == ()


def test_entry_with_no_cost_at_all_is_a_gap() -> None:
    found = catalog({"gpt-mystery": {"mode": "chat"}}).reconcile(
        inventory(LiveModel(provider="openai", model_id="gpt-mystery"))
    )
    assert [c.kind for c in found] == ["missing_price"]


def test_chat_entry_missing_output_cost_is_a_gap() -> None:
    found = catalog({"gpt-half": {"mode": "chat", "input_cost_per_token": 1e-06}}).reconcile(
        inventory(LiveModel(provider="openai", model_id="gpt-half"))
    )
    assert [c.kind for c in found] == ["missing_price"]


def test_model_absent_from_the_catalog_is_a_new_launch() -> None:
    found = catalog({}).reconcile(inventory(LiveModel(provider="openai", model_id="gpt-brand-new")))
    assert [c.kind for c in found] == ["new_launch"]


def test_rounding_difference_in_context_window_is_not_drift() -> None:
    """The cost map deliberately carries 65535 where providers report 65536."""
    found = catalog({"gemini/g": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06, "max_output_tokens": 65535}}).reconcile(
        inventory(LiveModel(provider="gemini", model_id="g", max_output_tokens=65536))
    )
    assert found == ()


def test_material_context_difference_is_drift() -> None:
    found = catalog({"gemini/g": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06, "max_input_tokens": 131072}}).reconcile(
        inventory(LiveModel(provider="gemini", model_id="g", max_input_tokens=1048576))
    )
    assert [c.kind for c in found] == ["context_drift"]


def test_published_shutdown_date_the_catalog_lacks_is_a_deprecation_signal() -> None:
    found = catalog({"gpt-old": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}}).reconcile(
        inventory(LiveModel(provider="openai", model_id="gpt-old", shutdown_date="2026-10-23"))
    )
    assert [c.kind for c in found] == ["deprecation_signal"]


def test_matching_shutdown_date_is_not_a_signal() -> None:
    found = catalog(
        {"gpt-old": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06, "deprecation_date": "2026-10-23"}}
    ).reconcile(inventory(LiveModel(provider="openai", model_id="gpt-old", shutdown_date="2026-10-23")))
    assert found == ()


def test_prefixed_providers_are_looked_up_under_their_prefix() -> None:
    assert catalog_keys_for(LiveModel(provider="gemini", model_id="g-1"))[0] == "gemini/g-1"
    assert catalog_keys_for(LiveModel(provider="openai", model_id="gpt-4")) == ("gpt-4",)


def test_bedrock_regional_variants_are_not_reported_as_new() -> None:
    """us./eu. prefixed copies of the same Bedrock model already exist in the catalog."""
    found = catalog({"us.anthropic.claude-x": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}}).reconcile(
        inventory(LiveModel(provider="bedrock", model_id="anthropic.claude-x"))
    )
    assert found == ()


def test_patch_excludes_candidates_that_failed_verification() -> None:
    unverified = PricedCandidate(
        candidate=catalog({}).reconcile(inventory(LiveModel(provider="openai", model_id="gpt-new")))[0],
        pricing=TokenPricing(),
    )
    assert build_patch(catalog({}), [unverified]).is_empty


def test_patch_adds_a_verified_new_model() -> None:
    candidate = catalog({}).reconcile(inventory(LiveModel(provider="openai", model_id="gpt-new")))[0]
    verified = PricedCandidate(
        candidate=candidate,
        pricing=TokenPricing(
            input_cost_per_token=Sourced(1e-06, provenance()),
            output_cost_per_token=Sourced(2e-06, provenance()),
        ),
    )
    patch = build_patch(catalog({}), [verified])
    assert patch.additions["gpt-new"]["input_cost_per_token"] == 1e-06
    assert patch.additions["gpt-new"]["litellm_provider"] == "openai"


def test_real_cost_map_loads_and_drops_the_sample_spec() -> None:
    path = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
    loaded = Catalog.load(path)
    assert "sample_spec" not in loaded.entries
    assert len(loaded.entries) == len(json.loads(path.read_text())) - 1
