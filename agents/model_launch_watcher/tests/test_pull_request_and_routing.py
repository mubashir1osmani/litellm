"""The pull request a delta produces, and how an inbound A2A message picks a skill."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from model_launch_watcher.card import (
    SKILL_DISCOVER_LAUNCHES,
    SKILL_DRIFT_AUDIT,
    SKILL_ISSUE_TRIAGE,
    SKILL_PR_SWEEP,
    SKILL_PRICE_DIFF,
    SKILL_RECORD_CORRECTION,
    build_agent_card,
)
from model_launch_watcher.domain import Candidate, Confidence, LiveModel, PriceDelta, PricedCandidate, Provenance
from model_launch_watcher.executor import route_providers, route_skill
from model_launch_watcher.jobs import parse_correction
from model_launch_watcher.memory import CorrectionKind
from model_launch_watcher.pull_request import apply_edits, render_pull_request

SOURCE = "https://docs.claude.com/pricing"


def drift_candidate() -> PricedCandidate:
    provenance = Provenance(
        source_url=SOURCE, retrieved_at=datetime(2026, 8, 20, tzinfo=UTC), confidence=Confidence.PRIMARY_DOC
    )
    return PricedCandidate(
        candidate=Candidate(
            kind="price_drift",
            live=LiveModel(provider="anthropic", model_id="claude-opus-5", display_name="Claude Opus 5"),
            summary="Claude Opus 5: input cut 40%",
            catalog_key="claude-opus-5",
            deltas=(
                PriceDelta(field="input_cost_per_token", catalogued=5e-06, live=3e-06, provenance=provenance),
            ),
        ),
        pricing=__import__("model_launch_watcher.domain", fromlist=["TokenPricing"]).TokenPricing(),
    )


def test_title_names_the_direction_of_the_change() -> None:
    draft = render_pull_request(drift_candidate(), "20260820")
    assert draft.title == "fix(pricing): sync claude-opus-5 with published price cut"


def test_branch_is_prefixed_and_contains_no_slash() -> None:
    """Repo convention: litellm_ prefixed branches, never a slash in the name."""
    draft = render_pull_request(drift_candidate(), "20260820")
    assert draft.branch.startswith("litellm_price_sync")
    assert "/" not in draft.branch


def test_body_carries_the_before_after_table_and_the_source() -> None:
    body = render_pull_request(drift_candidate(), "20260820").body
    assert "## Before / after" in body
    assert "| input_cost_per_token | 5.0000 | 3.0000 | -40.0% |" in body
    assert SOURCE in body
    assert "## Linear ticket" in body


def test_edits_carry_the_published_value_not_the_stale_one() -> None:
    assert render_pull_request(drift_candidate(), "20260820").edits == {"input_cost_per_token": 3e-06}


def test_apply_edits_preserves_other_fields_and_indent(tmp_path: Path) -> None:
    path = tmp_path / "cost_map.json"
    path.write_text(json.dumps({"claude-opus-5": {"input_cost_per_token": 5e-06, "mode": "chat"}}, indent=4))
    apply_edits(path, "claude-opus-5", {"input_cost_per_token": 3e-06})
    written = json.loads(path.read_text())
    assert written["claude-opus-5"] == {"input_cost_per_token": 3e-06, "mode": "chat"}
    assert '    "claude-opus-5"' in path.read_text()


def test_apply_edits_refuses_an_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "cost_map.json"
    path.write_text(json.dumps({}))
    with pytest.raises(KeyError):
        apply_edits(path, "not-there", {"input_cost_per_token": 1e-06})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("did any prices change?", SKILL_PRICE_DIFF),
        ("review the open pull requests", SKILL_PR_SWEEP),
        ("what pricing issues are stale?", SKILL_ISSUE_TRIAGE),
        ("run the weekly audit", SKILL_DRIFT_AUDIT),
        ("what shipped recently?", SKILL_DISCOVER_LAUNCHES),
    ],
)
def test_prose_routes_to_the_right_skill(text: str, expected: str) -> None:
    assert route_skill(text, {}) == expected


def test_declared_skill_beats_the_wording() -> None:
    assert route_skill("did any prices change?", {"skill": SKILL_PR_SWEEP}) == SKILL_PR_SWEEP


def test_unknown_declared_skill_falls_back_to_the_wording() -> None:
    assert route_skill("review the open pull requests", {"skill": "not_a_skill"}) == SKILL_PR_SWEEP


def test_a_correction_payload_routes_to_the_memory_skill() -> None:
    assert route_skill("anything", {"correction": {"kind": "map_name"}}) == SKILL_RECORD_CORRECTION


def test_providers_are_read_from_metadata_and_prose() -> None:
    assert route_providers("", {"providers": ["anthropic", "nonsense"]}) == ("anthropic",)
    assert route_providers("check gemini and openai", {}) == ("gemini", "openai")
    assert route_providers("check everything", {}) == ()


def test_correction_requires_structured_input() -> None:
    """Inferring a durable rule from prose is the kind of guess that needs correcting later."""
    assert isinstance(parse_correction("just remember it", {}, "sre"), str)


def test_correction_is_read_from_metadata() -> None:
    payload = {"kind": "map_name", "scope": "anthropic", "subject": "Claude X", "value": "claude-x", "reason": "alias"}
    parsed = parse_correction("", {"correction": payload}, "sre")
    assert not isinstance(parsed, str)
    assert parsed.kind is CorrectionKind.MAP_NAME
    assert parsed.value == "claude-x"


def test_correction_rejects_an_unknown_kind() -> None:
    payload = {"kind": "invent", "scope": "a", "subject": "b", "value": "c"}
    assert isinstance(parse_correction(json.dumps(payload), {}, "sre"), str)


def test_correction_rejects_missing_fields() -> None:
    payload = {"kind": "map_name", "scope": "anthropic"}
    outcome = parse_correction(json.dumps(payload), {}, "sre")
    assert isinstance(outcome, str)
    assert "subject" in outcome


def test_card_advertises_every_skill_the_router_can_reach() -> None:
    card = build_agent_card("http://localhost:8080")
    advertised = {skill.id for skill in card.skills}
    assert advertised == {
        SKILL_PRICE_DIFF,
        SKILL_DRIFT_AUDIT,
        SKILL_PR_SWEEP,
        SKILL_ISSUE_TRIAGE,
        SKILL_DISCOVER_LAUNCHES,
        SKILL_RECORD_CORRECTION,
    }
    assert card.supported_interfaces[0].url == "http://localhost:8080/"
