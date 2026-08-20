"""Durable memory of human corrections, so the agent is told a thing once.

The agent gets things wrong in recurring, specific ways: it maps a provider's display
name to the wrong catalog key, or it reports a value the team set deliberately. Without
memory each of those returns every week and the reports become noise people stop reading.

Corrections are append-only JSONL so the history is reviewable in git and a bad rule can
be traced to whoever added it. Later records win, which makes a correction revisable
without editing history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

from .domain import utc_now

GLOBAL_SCOPE: Final = "*"


class CorrectionKind(Enum):
    """What a human taught the agent.

    ``SUPPRESS`` stops a finding recurring. ``MAP_NAME`` binds a provider's published
    name, or an Azure meter, to a catalog key the agent could not infer. ``PIN_VALUE``
    marks a catalogued value as deliberate so drift checks leave it alone. ``CONVENTION``
    is guidance in prose, carried into the extraction prompt and into review comments.
    """

    SUPPRESS = "suppress"
    MAP_NAME = "map_name"
    PIN_VALUE = "pin_value"
    CONVENTION = "convention"


@dataclass(frozen=True, slots=True)
class Correction:
    kind: CorrectionKind
    scope: str
    subject: str
    value: str
    reason: str
    recorded_at: datetime
    recorded_by: str

    def to_json(self) -> Mapping[str, str]:
        return {
            "kind": self.kind.value,
            "scope": self.scope,
            "subject": self.subject,
            "value": self.value,
            "reason": self.reason,
            "recorded_at": self.recorded_at.isoformat(),
            "recorded_by": self.recorded_by,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, str]) -> Correction | None:
        try:
            kind: Final = CorrectionKind(row["kind"])
            recorded: Final = datetime.fromisoformat(row["recorded_at"])
        except (KeyError, ValueError):
            return None
        return cls(
            kind=kind,
            scope=row.get("scope", GLOBAL_SCOPE),
            subject=row.get("subject", ""),
            value=row.get("value", ""),
            reason=row.get("reason", ""),
            recorded_at=recorded,
            recorded_by=row.get("recorded_by", "unknown"),
        )


@dataclass(frozen=True, slots=True)
class Memory:
    """An immutable view of what the agent has been taught."""

    corrections: tuple[Correction, ...] = ()
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> Memory:
        if not path.exists():
            return cls(corrections=(), path=path)
        parsed: Final = tuple(
            correction
            for correction in (Correction.from_json(row) for row in _read_rows(path))
            if correction is not None
        )
        return cls(corrections=parsed, path=path)

    def record(self, correction: Correction) -> Memory:
        """Append a correction and return the updated memory. Writing is the point, so a
        memory with nowhere to write refuses rather than silently forgetting."""
        if self.path is None:
            raise ValueError("Memory has no path; cannot persist a correction")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(correction.to_json(), sort_keys=True) + "\n")
        return Memory(corrections=(*self.corrections, correction), path=self.path)

    def _matching(self, kind: CorrectionKind, scopes: Iterable[str]) -> tuple[Correction, ...]:
        allowed: Final = frozenset(scopes)
        return tuple(c for c in self.corrections if c.kind is kind and c.scope in allowed)

    def suppresses(self, catalog_key: str, provider: str, subject: str) -> Correction | None:
        """Return the correction that silences this finding, newest first, else ``None``."""
        scoped: Final = self._matching(CorrectionKind.SUPPRESS, (catalog_key, provider, GLOBAL_SCOPE))
        return next(
            (c for c in reversed(scoped) if c.subject in (subject, GLOBAL_SCOPE)),
            None,
        )

    def pinned(self, catalog_key: str, field: str) -> Correction | None:
        scoped: Final = self._matching(CorrectionKind.PIN_VALUE, (catalog_key,))
        return next((c for c in reversed(scoped) if c.subject == field), None)

    def mapped_key(self, provider: str, published_name: str) -> str | None:
        """Resolve a provider's published name to the catalog key a human bound it to."""
        scoped: Final = self._matching(CorrectionKind.MAP_NAME, (provider, GLOBAL_SCOPE))
        needle: Final = published_name.casefold().strip()
        return next((c.value for c in reversed(scoped) if c.subject.casefold().strip() == needle), None)

    def conventions(self, provider: str) -> tuple[str, ...]:
        scoped: Final = self._matching(CorrectionKind.CONVENTION, (provider, GLOBAL_SCOPE))
        return tuple(c.value for c in scoped)

    def summary(self) -> Mapping[str, int]:
        return {kind.value: sum(1 for c in self.corrections if c.kind is kind) for kind in CorrectionKind}


def _read_rows(path: Path) -> Sequence[Mapping[str, str]]:
    rows: Final = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return tuple(parsed for parsed in (_safe_load(r) for r in rows if r) if parsed is not None)


def _safe_load(line: str) -> Mapping[str, str] | None:
    try:
        loaded: Final = json.loads(line)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def correction(
    kind: CorrectionKind,
    scope: str,
    subject: str,
    value: str,
    reason: str,
    recorded_by: str,
) -> Correction:
    return Correction(
        kind=kind,
        scope=scope,
        subject=subject,
        value=value,
        reason=reason,
        recorded_at=utc_now(),
        recorded_by=recorded_by,
    )
