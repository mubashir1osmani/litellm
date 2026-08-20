"""The A2A agent card and the skills it advertises.

The skills are the four standing jobs, plus the one that makes the other four improve:
a caller can correct the agent, and the correction changes later runs.
"""

from __future__ import annotations

from typing import Final

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill
from a2a.utils import DEFAULT_RPC_URL, TransportProtocol

SKILL_PRICE_DIFF: Final = "price_diff"
SKILL_DRIFT_AUDIT: Final = "drift_audit"
SKILL_PR_SWEEP: Final = "pr_sweep"
SKILL_ISSUE_TRIAGE: Final = "issue_triage"
SKILL_DISCOVER_LAUNCHES: Final = "discover_launches"
SKILL_RECORD_CORRECTION: Final = "record_correction"

AGENT_NAME: Final = "Model Launch Watcher"

_SKILLS: Final[tuple[AgentSkill, ...]] = (
    AgentSkill(
        id=SKILL_PRICE_DIFF,
        name="Detect published price changes",
        description=(
            "Read each provider's official pricing page, extract the whole price table, and diff it "
            "against the catalogued per-token costs. Reports every model whose published price no "
            "longer matches the cost map, with a before/after table and the source URL. This is the "
            "check that catches a price cut, which otherwise goes unnoticed until someone is billed wrong."
        ),
        tags=["pricing", "drift", "monitor"],
        examples=["Did any prices change?", "Check anthropic and openai for price cuts"],
        input_modes=["text/plain"],
        output_modes=["application/json", "text/plain"],
    ),
    AgentSkill(
        id=SKILL_DRIFT_AUDIT,
        name="Full reconciliation audit",
        description=(
            "Reconcile the whole cost map against every reachable provider: price changes, context "
            "windows that materially disagree, published shutdown dates the map has not recorded, and "
            "entries carrying no cost at all. Reports a single diff count so the trend is visible "
            "week over week."
        ),
        tags=["audit", "reconciliation", "weekly"],
        examples=["Run the weekly audit", "How far out of date are we?"],
        input_modes=["text/plain"],
        output_modes=["application/json", "text/plain"],
    ),
    AgentSkill(
        id=SKILL_PR_SWEEP,
        name="Review open cost-map pull requests",
        description=(
            "Find every open pull request touching the cost map, read which model entries it changes, "
            "and check those entries against published provider prices. Returns a merge, close or "
            "needs-review recommendation per PR with the evidence behind it. Posts nothing by itself."
        ),
        tags=["github", "review", "community"],
        examples=["Review the open pricing PRs", "Which cost map PRs are safe to merge?"],
        input_modes=["text/plain"],
        output_modes=["application/json", "text/plain"],
    ),
    AgentSkill(
        id=SKILL_ISSUE_TRIAGE,
        name="Triage pricing issues",
        description=(
            "Sweep open issues touching pricing, suggest labels and an owner for each, and flag any "
            "that have gone more than 48 hours without a reply. Returns a digest ready to post to the "
            "team channel."
        ),
        tags=["github", "triage", "backlog"],
        examples=["What pricing issues need attention?", "Anything gone stale?"],
        input_modes=["text/plain"],
        output_modes=["application/json", "text/plain"],
    ),
    AgentSkill(
        id=SKILL_DISCOVER_LAUNCHES,
        name="Discover model launches",
        description=(
            "Ask the provider APIs what they serve and report models absent from the cost map. "
            "Discovery only, so it returns quickly and spends no tokens."
        ),
        tags=["discovery", "models"],
        examples=["What shipped that we don't know about?"],
        input_modes=["text/plain"],
        output_modes=["application/json", "text/plain"],
    ),
    AgentSkill(
        id=SKILL_RECORD_CORRECTION,
        name="Correct the agent",
        description=(
            "Teach the agent something it got wrong, and it will hold that correction on every later "
            "run. Bind a provider's published model name to a catalog key it could not infer, mark a "
            "catalogued value as deliberate so drift checks leave it alone, silence a finding, or add "
            "a convention that is carried into future price extraction."
        ),
        tags=["memory", "feedback", "corrections"],
        examples=[
            "Map 'Claude Haiku 3.5 (retired)' to claude-3-5-haiku-20241022",
            "We use 65535 on purpose, stop flagging it",
            "Never propose entries for preview models",
        ],
        input_modes=["text/plain", "application/json"],
        output_modes=["application/json", "text/plain"],
    ),
)

ALL_SKILL_IDS: Final[tuple[str, ...]] = tuple(skill.id for skill in _SKILLS)


def build_agent_card(public_url: str) -> AgentCard:
    return AgentCard(
        name=AGENT_NAME,
        description=(
            "Watches published LLM prices and keeps LiteLLM's cost map honest. Detects price changes "
            "against provider pages, audits the catalog, reviews community pricing pull requests and "
            "triages pricing issues. Every number it reports carries the URL it was read from and the "
            "time it was read, and it remembers corrections."
        ),
        version="0.1.0",
        documentation_url="https://docs.litellm.ai/docs/proxy/cost_tracking",
        provider=AgentProvider(organization="LiteLLM", url="https://github.com/BerriAI/litellm"),
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json", "text/plain"],
        skills=list(_SKILLS),
        supported_interfaces=[
            AgentInterface(
                url=f"{public_url.rstrip('/')}{DEFAULT_RPC_URL}",
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version="1.0",
            )
        ],
    )
