"""Deterministic short-term conversation compaction."""

from __future__ import annotations

from collections.abc import Iterable

from app.memory.canonicalizer import stable_unique
from app.schemas import ConversationSummary, ConversationTurn, Evidence


MAX_USER_GOALS = 20
MAX_USER_GOAL_LENGTH = 120


def _merge_mapping(original: dict[str, str], additions: dict[str, str]) -> dict[str, str]:
    merged = dict(original)
    merged.update(additions)
    return dict(sorted(merged.items()))


def _bounded_text(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= MAX_USER_GOAL_LENGTH:
        return normalized
    return f"{normalized[: MAX_USER_GOAL_LENGTH - 1].rstrip()}…"


def _turn_goal(turn: ConversationTurn) -> str:
    query = _bounded_text(turn.user)
    if turn.intent is None:
        return query
    prefix = f"{turn.intent.value}: "
    available = MAX_USER_GOAL_LENGTH - len(prefix)
    if len(query) > available:
        query = f"{query[: max(0, available - 1)].rstrip()}…"
    return f"{prefix}{query}" if query else turn.intent.value


def compact_turns(
    turns: list[ConversationTurn],
    summary: ConversationSummary | None,
    *,
    max_turns: int = 15,
    compact_oldest: int = 10,
    keep_recent: int = 5,
    resolved_entities: dict[str, str] | None = None,
    focus_entity_ids: list[str] | None = None,
    fact_ids: Iterable[str] = (),
    evidence: Iterable[Evidence] = (),
    latest_graph_id: str | None = None,
) -> tuple[list[ConversationTurn], ConversationSummary]:
    """Compact exactly at the configured boundary using replacement semantics."""

    current = summary or ConversationSummary()
    if len(turns) < max_turns:
        current.resolved_entities = _merge_mapping(
            current.resolved_entities, resolved_entities or {}
        )
        current.focus_entity_ids = stable_unique(
            [*current.focus_entity_ids, *(focus_entity_ids or [])]
        )
        current.confirmed_evidence_ids = stable_unique(
            [*current.confirmed_evidence_ids, *(item.id for item in evidence)]
        )
        current.confirmed_fact_ids = stable_unique(
            [*current.confirmed_fact_ids, *fact_ids]
        )
        if latest_graph_id:
            current.latest_graph_id = latest_graph_id
        return turns, current

    # A restored checkpoint may already be over the normal boundary.  Compact
    # every turn except the configured tail so no middle turn is silently lost.
    number_to_compact = max(compact_oldest, len(turns) - keep_recent)
    number_to_compact = min(number_to_compact, len(turns))
    compacted = turns[:number_to_compact]
    retained = turns[number_to_compact:]
    current.user_goals = stable_unique(
        [
            *(_bounded_text(goal) for goal in current.user_goals),
            *(_turn_goal(turn) for turn in compacted),
        ]
    )[-MAX_USER_GOALS:]
    current.resolved_entities = _merge_mapping(
        current.resolved_entities, resolved_entities or {}
    )
    current.focus_entity_ids = stable_unique(
        [
            *current.focus_entity_ids,
            *(entity_id for turn in compacted for entity_id in turn.focus_entity_ids),
            *(focus_entity_ids or []),
        ]
    )
    current.confirmed_evidence_ids = stable_unique(
        [*current.confirmed_evidence_ids, *(item.id for item in evidence)]
    )
    current.confirmed_fact_ids = stable_unique(
        [*current.confirmed_fact_ids, *fact_ids]
    )
    current.summarized_turns += len(compacted)
    if latest_graph_id:
        current.latest_graph_id = latest_graph_id
    return retained, current
