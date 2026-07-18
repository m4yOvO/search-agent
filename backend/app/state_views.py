"""Read-only semantic views over the request StateGraph state.

Planner semantics have one authoritative representation: ``planner_decision``.
After research or cache hydration the verified ``query_signature`` (and, for a
cache record, its ``CachedPayload``) carries the same durable semantics.  Keeping
copies such as ``intent`` or ``needs_clarification`` in independent StateGraph
channels makes those values able to drift, so consumers use this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from app.schemas import (
    CacheScope,
    CachedPayload,
    Intent,
    PlannerDecision,
    QuerySignature,
)


@dataclass(frozen=True, slots=True)
class RequestSemantics:
    """The non-factual query semantics needed by deterministic runtime nodes."""

    intent: Intent | None
    needs_clarification: bool
    clarification_question: str | None
    query_requires_realtime_data: bool
    cache_scope: CacheScope


def _validated_model(
    value: Any,
    model_type: type[PlannerDecision] | type[QuerySignature] | type[CachedPayload],
) -> PlannerDecision | QuerySignature | CachedPayload | None:
    if value is None:
        return None
    try:
        return model_type.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None


def planner_decision(state: Mapping[str, Any]) -> PlannerDecision | None:
    value = _validated_model(state.get("planner_decision"), PlannerDecision)
    return value if isinstance(value, PlannerDecision) else None


def query_signature(state: Mapping[str, Any]) -> QuerySignature | None:
    value = _validated_model(state.get("query_signature"), QuerySignature)
    if isinstance(value, QuerySignature):
        return value
    payload = cached_payload(state)
    return payload.query_signature if payload is not None else None


def cached_payload(state: Mapping[str, Any]) -> CachedPayload | None:
    lookup = state.get("cache_lookup")
    if isinstance(lookup, Mapping):
        if not lookup.get("hit"):
            return None
        value = lookup.get("payload")
    else:
        if not getattr(lookup, "hit", False):
            return None
        value = getattr(lookup, "payload", None)
    parsed = _validated_model(value, CachedPayload)
    return parsed if isinstance(parsed, CachedPayload) else None


def request_semantics(state: Mapping[str, Any]) -> RequestSemantics:
    """Derive request semantics without consulting duplicated state channels.

    A Planner decision is authoritative while the fresh path is running.  A raw
    cache hit intentionally has no Planner call, so its verified signature and
    payload are the fallback.  Cache scope is durable payload/signature data and
    is therefore derived independently instead of copied through the graph.
    """

    plan = planner_decision(state)
    signature = query_signature(state)
    payload = cached_payload(state)

    intent = (
        plan.intent
        if plan is not None
        else signature.intent
        if signature is not None
        else None
    )
    needs_clarification = bool(plan is not None and plan.intent is Intent.CLARIFY)
    clarification_question = (
        plan.clarification_question if needs_clarification and plan is not None else None
    )
    query_requires_realtime_data = bool(
        plan is not None and plan.query_requires_realtime_data
    )

    if payload is not None:
        cache_scope = payload.cache_scope
    elif signature is not None:
        cache_scope = (
            CacheScope.CONVERSATION
            if signature.context_entity_ids
            else CacheScope.CONTEXT_FREE
        )
    elif plan is not None:
        cache_scope = (
            CacheScope.CONVERSATION
            if any(
                reference.context_entity_id is not None
                for reference in plan.entity_references
            )
            else CacheScope.CONTEXT_FREE
        )
    else:
        cache_scope = CacheScope.CONTEXT_FREE

    return RequestSemantics(
        intent=intent,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        query_requires_realtime_data=query_requires_realtime_data,
        cache_scope=cache_scope,
    )
