# Model Launch Watcher

An A2A agent that keeps `model_prices_and_context_window.json` honest. It reads what providers
actually publish, diffs that against the cost map, and proposes changes with the source URL
attached. It never edits the cost map on its own.

The highest-value thing it does is catch a **price cut**. Everything else in the repo's tooling
notices models we are missing; nothing noticed a model whose price changed under us, which is
the case that quietly bills every caller at the wrong rate.

## The four jobs

| Job | What it does | Schedule |
| --- | --- | --- |
| `price_diff` | Reads each provider's pricing page, extracts the whole price table, diffs it against the catalogued per-token costs, and drafts one PR per delta with a before/after table and the source URL | daily |
| `drift_audit` | Reconciles everything reachable: price changes, context windows that materially disagree, published shutdown dates we have not recorded, entries with no cost at all. Reports a diff count and the coverage it was drawn from | weekly |
| `pr_sweep` | Finds open PRs touching the cost map, reads which entries each one changes, validates those against published prices, and returns merge / close / needs-review with the evidence | daily |
| `issue_triage` | Sweeps open pricing issues, suggests labels and an owner, flags anything with no reply past 48h, and builds a digest for the team channel | every morning |

## Coverage, and why the number is small

Every report states its own denominator, because a drift count without one reads as a
clean bill of health. A real unscoped run today:

```
Providers reached: anthropic, gemini, mistral, openai, vertex_ai, xai
87 findings (1 price changes, 16 new, 0 unpriced, 54 deprecations, 16 context drift)
Price coverage: 17 of 2410 token-billed entries were compared against a published price (0.7%)
```

Price coverage is low and will stay low, for a structural reason. A provider's pricing page
lists its current flagship models, roughly fifteen of them, while the cost map carries 2410
token-billed entries dominated by aggregators and hosts (fireworks, novita, deepinfra) that
publish no first-party price page this agent can read. Treat price coverage as "the models
providers actively publish", not as the catalog.

Deprecation and launch coverage are much better, because those come from discovery APIs that
list everything a provider serves. The 54 deprecation findings above are real: OpenAI returns
a `shutdown_date` per model and the cost map has not recorded most of them.

## Why you can trust a number it reports

Provider *discovery* has APIs behind it. Provider *pricing* mostly does not, so prices are read
off published pages by an LLM, which makes hallucination the central risk. Three things contain it.

**Verbatim grounding.** The extractor must return the snippet it read each number from. That
snippet has to appear character-for-character in the fetched page, and the number has to appear
inside the snippet. Anything else is dropped and reported as a gap. A provider restyling its
pricing table makes this agent go quiet, never confidently wrong.

**Provenance on every value.** Each price carries the URL, the retrieval timestamp, and a
confidence tier: `primary_api` (a structured field from the provider's own API), `primary_doc`
(grounded in the provider's published page), or `aggregator` (a third party restating them).
An aggregator can corroborate a price. It can never establish one.

**No name guessing.** A published name is bound to a catalog key only when the mapping is exact
or a human taught it. `Claude Haiku 3.5 (retired, except on Bedrock…)` maps to nothing on its own,
so it is reported as needing a mapping rather than matched approximately. Binding a price to the
wrong model is worse than reporting nothing.

## Memory

Correct it once and it holds the correction. Corrections live in append-only JSONL, so the
history is reviewable in git and a bad rule traces back to whoever added it.

| Kind | Use it for |
| --- | --- |
| `map_name` | Bind a published name or an Azure meter to a catalog key it could not infer |
| `pin_value` | Mark a catalogued value as deliberate, so drift checks leave it alone |
| `suppress` | Silence a finding that is tracked elsewhere |
| `convention` | Prose guidance carried into future price extraction |

```bash
curl -s localhost:8080/ -H 'content-type: application/json' -H 'A2A-Version: 1.0' -d '{
  "jsonrpc":"2.0","id":1,"method":"SendMessage",
  "params":{"message":{"messageId":"m1","role":"ROLE_USER","metadata":{"skill":"record_correction"},
  "parts":[{"text":"{\"kind\":\"map_name\",\"scope\":\"anthropic\",\"subject\":\"Claude Haiku 3.5 ( retired, except on Bedrock and Google Cloud )\",\"value\":\"anthropic.claude-3-5-haiku-20241022-v1:0\",\"reason\":\"page appends retirement notes to the display name\"}"}]}}}'
```

The `A2A-Version: 1.0` header is required; without it the SDK assumes 0.3 and refuses the call.

`pin_value` is what the `65535` versus `65536` case is for: the cost map carries 65535 on purpose,
so pin it and the weekly audit stops re-reporting it.

## Running it

```bash
uv venv && uv pip install -e '.[dev]'

model-launch-watcher price_diff --provider anthropic     # drafts PRs, opens nothing
model-launch-watcher drift_audit                          # everything reachable, with coverage
model-launch-watcher pr_sweep                             # validate open community PRs
model-launch-watcher issue_triage --post                  # triage and deliver the digest
```

`price_diff` always prints the pull requests it would open. `--open-pr` is what actually opens
them, and it is off by default because it writes to a public repository. When it runs it edits
both copies of the cost map (CI fails if they diverge), regenerates the schema via
`ci_cd/generate_model_prices_schema.py`, branches `litellm_price_sync_*`, and targets
`litellm_internal_staging`.

Exit status is 0 whenever the agent could see something. Findings are the normal output of a
healthy run and never fail the job. A non-zero exit means every source was unreachable, which
is the only condition a cron should alert on.

### Credentials

Everything degrades to a partial run rather than an error. A provider with no key is reported as
a gap and the rest still runs.

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY` | model discovery |
| `AWS_ACCESS_KEY_ID` / `AWS_PROFILE` | Bedrock `ListFoundationModels` |
| `GITHUB_TOKEN`, else the `gh` CLI's token | PR sweep and issue triage |
| `SLACK_WEBHOOK_URL` | the triage digest, otherwise it prints to stdout |
| `WATCHER_MEMORY_PATH` | corrections file, defaults to `corrections.jsonl` beside this README |

## As an A2A agent

```bash
model-launch-watcher-serve                       # or: uvicorn model_launch_watcher.server:create_app --factory
curl localhost:8080/.well-known/agent-card.json
```

The card advertises six skills: the four jobs, plus `discover_launches` and `record_correction`.
A caller names one in message metadata (`{"skill": "price_diff"}`) or just asks in prose and gets
routed. Providers can be narrowed the same way, so an agent that only cares about Bedrock does
not wait on every other provider's page.

The proxy in this repo can register this URL as an upstream A2A agent, which puts the watcher
behind the same auth, budgets and logging as everything else.

## Known gaps

These are real and deliberately visible rather than papered over.

- **Azure** publishes a genuinely machine-readable price API (`prices.azure.com/api/retail/prices`),
  but its meter names are abbreviated beyond safe inference: `gpt 4.1 Inp regnl Tokens`,
  `5.4 opt Dz 1M Tokens`. Mapping those automatically is a price-corruption risk pointed at the
  file that bills customers, so Azure meters go through `map_name` corrections. Note also that
  `unitOfMeasure` varies between `1K` and `1M` on that API, so it must be read per row.
- **Groq** and some others serve client-rendered pricing pages. The agent reports `doc_unusable`
  rather than extracting from a near-empty document.
- Providers with no published price page (fireworks, novita, moonshot and friends) mean a
  community PR adding them lands as `needs_review`, not `merge`. That is intended: `close` is
  reserved for a price the agent can actively contradict.

## Layout

```
domain.py       value types; provenance, confidence, deltas. Nothing here raises
catalog.py      reads the cost map, reconciles an inventory against it, builds the patch
discovery.py    provider-native model discovery, one adapter per API shape
pricing.py      page fetch, whole-table extraction, and the verbatim grounding check
price_diff.py   published price versus catalogued price, and name resolution
corroboration.py  aggregator cross-check that can agree but never establish
memory.py       durable corrections
graph.py        the LangGraph pipeline wiring the above together
github.py       the GitHub calls the two sweep jobs need
sweep.py        PR validation and issue triage
pull_request.py before/after PR rendering and the gated open path
card.py         A2A card and skills
executor.py     A2A request to job routing
jobs.py         one entry point per skill, shared by A2A and the CLI
```

The graph is deterministic and the LLM appears at exactly one node, reading a fetched document.
The output edits pricing that bills real callers, so the run has to be reproducible and every
number has to trace to a URL. A free-roaming agent loop would buy flexibility this task does not want.
