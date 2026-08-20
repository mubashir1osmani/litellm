"""The PR sweep and issue triage jobs, driven through injected fakes rather than the network."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

from model_launch_watcher.catalog import Catalog
from model_launch_watcher.domain import Confidence, Provenance, SourceFailure, Sourced, TokenPricing, utc_now
from model_launch_watcher.github import Issue, PullRequest, cost_map_keys_in_diff, touches_cost_map
from model_launch_watcher.memory import Memory
from model_launch_watcher.pricing import ExtractedTable, PricingDoc
from model_launch_watcher.sweep import (
    KeyVerdict,
    review_cost_map_prs,
    triage_digest,
    triage_pricing_issues,
)

ADD_BLOCK_DIFF = '''diff --git a/model_prices_and_context_window.json b/model_prices_and_context_window.json
--- a/model_prices_and_context_window.json
+++ b/model_prices_and_context_window.json
@@ -100,6 +100,10 @@
     "existing-model": {
         "mode": "chat"
     },
+    "moonshot/kimi-k3": {
+        "input_cost_per_token": 1e-06,
+        "output_cost_per_token": 2e-06
+    },
'''

MODIFY_FIELD_DIFF = '''diff --git a/litellm/model_prices_and_context_window_backup.json b/litellm/model_prices_and_context_window_backup.json
--- a/litellm/model_prices_and_context_window_backup.json
+++ b/litellm/model_prices_and_context_window_backup.json
@@ -740,6 +740,7 @@
     "qwen.qwen3-235b-v1:0": {
         "input_cost_per_token": 1e-06,
+        "input_cost_per_token_batches": 5e-07,
         "mode": "chat",
         "search_context_cost_per_query": {
+            "search_context_size_low": 0.01
         }
     },
'''

UNRELATED_DIFF = '''diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
'''


def test_added_block_is_detected() -> None:
    assert cost_map_keys_in_diff(ADD_BLOCK_DIFF) == ("moonshot/kimi-k3",)


def test_field_edit_inside_an_existing_entry_is_attributed_to_that_entry() -> None:
    """A price change edits a field, not a block. These are the PRs most worth validating."""
    assert cost_map_keys_in_diff(MODIFY_FIELD_DIFF) == ("qwen.qwen3-235b-v1:0",)


def test_nested_objects_are_not_mistaken_for_models() -> None:
    assert "search_context_cost_per_query" not in cost_map_keys_in_diff(MODIFY_FIELD_DIFF)


def test_the_mirrored_backup_file_counts_as_the_cost_map() -> None:
    """ci_cd/check_files_match.py keeps both copies identical, so either one is a cost-map edit."""
    assert touches_cost_map(MODIFY_FIELD_DIFF)


def test_unrelated_diff_is_ignored() -> None:
    assert not touches_cost_map(UNRELATED_DIFF)
    assert cost_map_keys_in_diff(UNRELATED_DIFF) == ()


@dataclass(frozen=True, slots=True)
class FakeGitHub:
    pulls: tuple[PullRequest, ...] = ()
    diffs: tuple[tuple[int, str], ...] = ()
    issues: tuple[Issue, ...] = ()

    async def open_pull_requests(self, repo: str, limit: int) -> Sequence[PullRequest]:
        return self.pulls

    async def pull_request_diff(self, repo: str, number: int) -> str:
        return dict(self.diffs).get(number, "")

    async def open_issues(self, repo: str, limit: int) -> Sequence[Issue]:
        return self.issues


@dataclass(frozen=True, slots=True)
class FakeExtractor:
    """Returns a fixed grounded table, so the sweep is tested without a network or an LLM."""

    priced_name: str = "Moonshot Kimi K3"

    async def extract_table(self, doc: PricingDoc, conventions: Sequence[str]) -> ExtractedTable:
        return ExtractedTable(doc=doc, rows=())


def pull_request(number: int, title: str) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        author="contributor",
        url=f"https://github.com/BerriAI/litellm/pull/{number}",
        updated_at=utc_now(),
        touched_cost_map=False,
    )


async def test_sweep_recommends_review_when_no_page_prices_the_key() -> None:
    """Most community PRs add providers with no machine-readable price page.

    Defaulting those to 'close' would reject exactly the contributions worth keeping.
    """
    github = FakeGitHub(pulls=(pull_request(1, "add kimi"),), diffs=((1, ADD_BLOCK_DIFF),))
    reviews, _ = await review_cost_map_prs(
        github, Catalog(entries=MappingProxyType({})), FakeExtractor(), Memory(), "owner/repo"
    )
    assert [r.recommendation for r in reviews] == ["needs_review"]
    assert reviews[0].pull_request.changed_keys == ("moonshot/kimi-k3",)


async def test_sweep_ignores_pull_requests_that_do_not_touch_the_cost_map() -> None:
    github = FakeGitHub(pulls=(pull_request(2, "docs"),), diffs=((2, UNRELATED_DIFF),))
    reviews, _ = await review_cost_map_prs(
        github, Catalog(entries=MappingProxyType({})), FakeExtractor(), Memory(), "owner/repo"
    )
    assert reviews == ()


async def test_sweep_surfaces_a_github_failure_instead_of_raising() -> None:
    @dataclass(frozen=True, slots=True)
    class BrokenGitHub:
        async def open_pull_requests(self, repo: str, limit: int) -> SourceFailure:
            return SourceFailure(source="github", reason="http_403", detail="rate limited")

        async def pull_request_diff(self, repo: str, number: int) -> str:
            return ""

        async def open_issues(self, repo: str, limit: int) -> Sequence[Issue]:
            return ()

    reviews, failures = await review_cost_map_prs(
        BrokenGitHub(), Catalog(entries=MappingProxyType({})), FakeExtractor(), Memory(), "owner/repo"
    )
    assert reviews == ()
    assert failures[0].reason == "http_403"


def issue(number: int, title: str, hours_old: float, comments: int, labels: tuple[str, ...] = ()) -> Issue:
    return Issue(
        number=number,
        title=title,
        author="reporter",
        url=f"https://github.com/BerriAI/litellm/issues/{number}",
        created_at=datetime.now(UTC) - timedelta(hours=hours_old),
        labels=labels,
        comment_count=comments,
    )


def test_triage_selects_only_pricing_issues() -> None:
    triaged = triage_pricing_issues((issue(1, "[Bug]: missing pricing for gpt-5.6", 1, 0), issue(2, "UI button broken", 1, 0)))
    assert [t.issue.number for t in triaged] == [1]


def test_issue_with_no_reply_past_48h_is_nudged() -> None:
    triaged = triage_pricing_issues((issue(1, "wrong cost for claude", 72, 0),))
    assert triaged[0].needs_nudge


def test_answered_issue_is_not_nudged() -> None:
    triaged = triage_pricing_issues((issue(1, "wrong cost for claude", 72, 3),))
    assert not triaged[0].needs_nudge


def test_recent_unanswered_issue_is_not_nudged_yet() -> None:
    triaged = triage_pricing_issues((issue(1, "wrong cost for claude", 5, 0),))
    assert not triaged[0].needs_nudge


def test_owner_is_suggested_from_an_existing_label() -> None:
    triaged = triage_pricing_issues((issue(1, "pricing bug", 1, 0, labels=("llm translation",)),))
    assert triaged[0].suggested_owner == "llm-translation"


def test_labels_already_present_are_not_suggested_again() -> None:
    triaged = triage_pricing_issues((issue(1, "pricing bug", 1, 0, labels=("pricing",)),))
    assert "pricing" not in triaged[0].suggested_labels


def test_digest_reports_the_stale_count() -> None:
    digest = triage_digest(triage_pricing_issues((issue(1, "pricing bug", 72, 0), issue(2, "cost wrong", 1, 0))), "o/r")
    assert "2 open pricing issues, 1 with no reply past 48h" in digest


def test_empty_digest_says_so() -> None:
    assert "No open pricing issues" in triage_digest((), "o/r")
