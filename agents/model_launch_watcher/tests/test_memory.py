"""Memory has to survive a restart and let a later correction override an earlier one."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_launch_watcher.memory import GLOBAL_SCOPE, CorrectionKind, Memory, correction


def fresh(tmp_path: Path) -> Memory:
    return Memory.load(tmp_path / "corrections.jsonl")


def test_a_correction_survives_a_reload(tmp_path: Path) -> None:
    """The whole point of memory: told once, held afterwards."""
    path = tmp_path / "corrections.jsonl"
    Memory.load(path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Claude X", "claude-x", "alias", "sre")
    )
    assert Memory.load(path).mapped_key("anthropic", "Claude X") == "claude-x"


def test_a_later_correction_overrides_an_earlier_one(tmp_path: Path) -> None:
    memory = fresh(tmp_path)
    memory = memory.record(correction(CorrectionKind.MAP_NAME, "anthropic", "Claude X", "wrong-key", "first", "sre"))
    memory = memory.record(correction(CorrectionKind.MAP_NAME, "anthropic", "Claude X", "right-key", "fixed", "sre"))
    assert memory.mapped_key("anthropic", "Claude X") == "right-key"


def test_name_lookup_ignores_case_and_padding(tmp_path: Path) -> None:
    memory = fresh(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Claude X", "claude-x", "alias", "sre")
    )
    assert memory.mapped_key("anthropic", "  claude x  ") == "claude-x"


def test_a_mapping_does_not_leak_across_providers(tmp_path: Path) -> None:
    memory = fresh(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, "anthropic", "Claude X", "claude-x", "alias", "sre")
    )
    assert memory.mapped_key("openai", "Claude X") is None


def test_global_scope_applies_everywhere(tmp_path: Path) -> None:
    memory = fresh(tmp_path).record(
        correction(CorrectionKind.MAP_NAME, GLOBAL_SCOPE, "Shared Name", "shared-key", "alias", "sre")
    )
    assert memory.mapped_key("bedrock", "Shared Name") == "shared-key"


def test_suppression_matches_provider_scope(tmp_path: Path) -> None:
    memory = fresh(tmp_path).record(
        correction(CorrectionKind.SUPPRESS, "gemini", "context_drift", "", "known upstream quirk", "sre")
    )
    assert memory.suppresses("gemini/gemini-3-pro", "gemini", "context_drift") is not None
    assert memory.suppresses("gemini/gemini-3-pro", "gemini", "price_drift") is None


def test_conventions_reach_the_extraction_prompt(tmp_path: Path) -> None:
    memory = fresh(tmp_path).record(
        correction(CorrectionKind.CONVENTION, "openai", "batch", "Ignore batch pricing rows", "we bill live", "sre")
    )
    assert memory.conventions("openai") == ("Ignore batch pricing rows",)


def test_a_corrupt_line_does_not_lose_the_rest(tmp_path: Path) -> None:
    """A half-written line from a crashed run must not blank the agent's memory."""
    path = tmp_path / "corrections.jsonl"
    Memory.load(path).record(correction(CorrectionKind.MAP_NAME, "anthropic", "A", "a", "r", "sre"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    Memory.load(path).record(correction(CorrectionKind.MAP_NAME, "anthropic", "B", "b", "r", "sre"))
    reloaded = Memory.load(path)
    assert reloaded.mapped_key("anthropic", "A") == "a"
    assert reloaded.mapped_key("anthropic", "B") == "b"


def test_recording_without_a_path_refuses_rather_than_forgetting(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Memory().record(correction(CorrectionKind.MAP_NAME, "anthropic", "A", "a", "r", "sre"))


def test_summary_counts_by_kind(tmp_path: Path) -> None:
    memory = fresh(tmp_path)
    memory = memory.record(correction(CorrectionKind.MAP_NAME, "anthropic", "A", "a", "r", "sre"))
    memory = memory.record(correction(CorrectionKind.SUPPRESS, "anthropic", "price_drift", "", "r", "sre"))
    assert memory.summary()["map_name"] == 1
    assert memory.summary()["suppress"] == 1
