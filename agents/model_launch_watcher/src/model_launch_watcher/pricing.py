"""Pricing extraction from provider-published documents, with verbatim grounding.

Providers publish prices as prose and tables, not as APIs, so this step reads their
official pages and uses an LLM to locate the numbers. That makes hallucination the
central risk, so an extracted price is only believed when the model also returns the
snippet it read the number from and that snippet is found verbatim in the fetched
document, with the number present inside the snippet.

The design bias is to emit nothing. A provider that restyles its pricing table should
cause this module to go quiet and report a gap, never to emit a confident wrong price.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Mapping, Protocol, Sequence

import httpx

from .domain import Confidence, Provenance, SourceFailure, Sourced, TokenPricing, utc_now

_TAG_SOUP: Final = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAGS: Final = re.compile(r"<[^>]+>")
_WHITESPACE: Final = re.compile(r"\s+")
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")

_PER_MILLION: Final = 1_000_000.0

PRICING_PAGES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai": "https://platform.openai.com/docs/pricing",
        "anthropic": "https://docs.claude.com/en/docs/about-claude/pricing",
        "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
        "vertex_ai": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "bedrock": "https://aws.amazon.com/bedrock/pricing/",
        "mistral": "https://mistral.ai/pricing",
        "xai": "https://docs.x.ai/docs/models",
        "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
        "groq": "https://groq.com/pricing",
    }
)


@dataclass(frozen=True, slots=True)
class PricingDoc:
    provider: str
    url: str
    text: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class ExtractedQuote:
    """What the LLM claims it read, paired with the snippet it claims to have read it from."""

    input_per_million: float
    output_per_million: float
    input_quote: str
    output_quote: str
    cache_read_per_million: float | None = None
    cache_read_quote: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedRow(ExtractedQuote):
    """One row of a whole-page extraction, carrying the name the provider published."""

    published_name: str = ""


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    """Every model priced on one page, from a single extraction call.

    Reading the page once per provider rather than once per model is what makes a full
    catalog reconciliation affordable: the entire Anthropic price list costs roughly ten
    thousand tokens, against several hundred calls for the per-model approach.
    """

    doc: PricingDoc
    rows: tuple[ExtractedRow, ...]


class TableExtractor(Protocol):
    async def extract_table(self, doc: PricingDoc, conventions: Sequence[str]) -> ExtractedTable | SourceFailure: ...


def html_to_text(html: str) -> str:
    return _WHITESPACE.sub(" ", _TAGS.sub(" ", _TAG_SOUP.sub(" ", html))).strip()


async def fetch_pricing_doc(provider: str, url: str, client: httpx.AsyncClient) -> PricingDoc | SourceFailure:
    try:
        response: Final = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; litellm-model-watcher)"})
    except httpx.HTTPError as exc:
        return SourceFailure(source=provider, reason="unreachable", detail=f"{url}: {type(exc).__name__}: {exc}")
    if response.status_code != httpx.codes.OK:
        return SourceFailure(source=provider, reason=f"http_{response.status_code}", detail=url)
    text: Final = html_to_text(response.text)
    if len(text) < _MIN_USEFUL_CHARS:
        return SourceFailure(
            source=provider,
            reason="doc_unusable",
            detail=f"{url}: {len(text)} chars of text, likely client-rendered",
        )
    return PricingDoc(provider=provider, url=url, text=text, retrieved_at=utc_now())


_MIN_USEFUL_CHARS: Final = 2_000


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).casefold().strip()


def _value_appears_in(quote: str, value: float) -> bool:
    """Guard against a model quoting a real snippet but reporting a number absent from it."""
    found: Final = _NUMBER.findall(quote.replace(",", ""))
    return any(math.isclose(float(n), value, rel_tol=1e-9) for n in found)


def ground(doc: PricingDoc, quote: str | None, per_million: float | None) -> Sourced[float] | None:
    """Convert a claimed price to a per-token cost only if the document really says it.

    Returns ``None`` on any doubt: no quote, a quote absent from the document, or a
    number that does not occur inside the quote it supposedly came from.
    """
    if quote is None or per_million is None or per_million < 0:
        return None
    if _normalize(quote) not in _normalize(doc.text):
        return None
    if not _value_appears_in(quote, per_million):
        return None
    return Sourced(
        value=per_million / _PER_MILLION,
        provenance=Provenance(source_url=doc.url, retrieved_at=doc.retrieved_at, confidence=Confidence.PRIMARY_DOC),
    )


def ground_all(doc: PricingDoc, quote: ExtractedQuote) -> TokenPricing:
    return TokenPricing(
        input_cost_per_token=ground(doc, quote.input_quote, quote.input_per_million),
        output_cost_per_token=ground(doc, quote.output_quote, quote.output_per_million),
        cache_read_input_token_cost=ground(doc, quote.cache_read_quote, quote.cache_read_per_million),
    )


def _first_text(response: object) -> str | None:
    choices: Final = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or not choices:
        return None
    content: Final = getattr(getattr(choices[0], "message", None), "content", None)
    return content if isinstance(content, str) and content.strip() else None


def _strip_code_fence(content: str) -> str:
    trimmed: Final = content.strip()
    if not trimmed.startswith("```"):
        return trimmed
    body: Final = trimmed.removeprefix("```json").removeprefix("```").removesuffix("```")
    return body.strip()


_TABLE_PROMPT: Final = """Extract EVERY model priced on this provider pricing page.

Return a single JSON object and nothing else:
  {{"models": [
    {{"published_name": "the model name exactly as the page writes it",
      "input_per_million": number, "output_per_million": number,
      "input_quote": "exact substring stating the input price",
      "output_quote": "exact substring stating the output price",
      "cache_read_per_million": number or null,
      "cache_read_quote": "exact substring" or null}}
  ]}}

Rules:
- Every quote must be copied character for character from the page, under 120 characters,
  and must contain the number you reported next to it.
- Prices must be per one million tokens. If the page quotes per 1K tokens, skip the row
  rather than converting it.
- Skip any model whose price you cannot quote exactly. A short, correct list is the goal.
- Do not invent models and do not carry a price across from a similarly named model.
{conventions}
PAGE:
{page}
"""


def _conventions_block(conventions: Sequence[str]) -> str:
    if not conventions:
        return ""
    lines: Final = "\n".join(f"- {c}" for c in conventions)
    return f"\nTeam conventions you have been given, follow them exactly:\n{lines}\n"


@dataclass(frozen=True, slots=True)
class LiteLLMTableExtractor:
    """Whole-page extraction through LiteLLM, so the cost map is maintained by the gateway it feeds."""

    model: str = "claude-sonnet-5"
    max_page_chars: int = 120_000

    async def extract_table(self, doc: PricingDoc, conventions: Sequence[str]) -> ExtractedTable | SourceFailure:
        from litellm import acompletion

        prompt: Final = _TABLE_PROMPT.format(
            conventions=_conventions_block(conventions), page=doc.text[: self.max_page_chars]
        )
        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            return SourceFailure(source=doc.provider, reason="extractor_error", detail=f"{type(exc).__name__}: {exc}")
        content: Final = _first_text(response)
        if content is None:
            return SourceFailure(source=doc.provider, reason="extractor_empty", detail=doc.url)
        return parse_table(content, doc)


def parse_table(content: str, doc: PricingDoc) -> ExtractedTable | SourceFailure:
    try:
        payload: Final = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        return SourceFailure(source=doc.provider, reason="extractor_malformed", detail=str(exc))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return SourceFailure(source=doc.provider, reason="extractor_malformed", detail="no models array")
    return ExtractedTable(doc=doc, rows=tuple(r for r in (_parse_row(m) for m in payload["models"]) if r is not None))


def _parse_row(raw: object) -> ExtractedRow | None:
    if not isinstance(raw, dict):
        return None
    name: Final = raw.get("published_name")
    numbers: Final = (raw.get("input_per_million"), raw.get("output_per_million"))
    quotes: Final = (raw.get("input_quote"), raw.get("output_quote"))
    if not isinstance(name, str) or not name.strip():
        return None
    if not all(isinstance(n, (int, float)) for n in numbers) or not all(isinstance(q, str) for q in quotes):
        return None
    cache_price: Final = raw.get("cache_read_per_million")
    cache_quote: Final = raw.get("cache_read_quote")
    return ExtractedRow(
        published_name=name.strip(),
        input_per_million=float(numbers[0]),
        output_per_million=float(numbers[1]),
        input_quote=str(quotes[0]),
        output_quote=str(quotes[1]),
        cache_read_per_million=float(cache_price) if isinstance(cache_price, (int, float)) else None,
        cache_read_quote=cache_quote if isinstance(cache_quote, str) else None,
    )


def ground_table(table: ExtractedTable) -> Mapping[str, TokenPricing]:
    """Ground every extracted row, keyed by published name. Ungrounded rows are dropped."""
    grounded: Final = ((row.published_name, ground_all(table.doc, row)) for row in table.rows)
    return {name: pricing for name, pricing in grounded if pricing.is_complete}
