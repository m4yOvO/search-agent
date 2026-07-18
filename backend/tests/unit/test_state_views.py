from __future__ import annotations

from app.memory.chroma_store import CacheLookup
from app.schemas import (
    CacheScope,
    CachedPayload,
    GraphPayload,
    Intent,
    PlannerDecision,
    QuerySignature,
)
from app.state_views import request_semantics


def test_request_semantics_uses_planner_without_duplicate_state_fields() -> None:
    question = "请明确你指的是哪一家同名企业？"
    decision = PlannerDecision.model_validate(
        {
            "intent": "clarify",
            "entity_references": [],
            "research_tasks": [],
            "result_merge": "not_applicable",
            "clarification_question": question,
            "query_requires_realtime_data": False,
        }
    )

    semantics = request_semantics({"planner_decision": decision})

    assert semantics.intent is Intent.CLARIFY
    assert semantics.needs_clarification is True
    assert semantics.clarification_question == question
    assert semantics.query_requires_realtime_data is False
    assert semantics.cache_scope is CacheScope.CONTEXT_FREE


def test_request_semantics_supports_raw_cache_hit_without_planner() -> None:
    signature = QuerySignature(
        intent=Intent.LOCATE_ENTITIES,
        subject_ids=["company:fictional"],
        context_entity_ids=["company:fictional"],
        relation_types=["headquartered_in"],
        requested_relation_types=["headquartered_in"],
        effective_relation_types=["headquartered_in"],
        target_types=["location"],
    )
    graph = GraphPayload(
        graph_id="graph:fictional-cache",
        data_version="fictional-v1",
    )
    payload = CachedPayload(
        answer="虚构缓存回答。",
        graph=graph,
        evidence=[],
        query_signature=signature,
        cache_scope=CacheScope.CONVERSATION,
    )

    semantics = request_semantics(
        {"cache_lookup": CacheLookup(hit=True, payload=payload)}
    )

    assert semantics.intent is Intent.LOCATE_ENTITIES
    assert semantics.needs_clarification is False
    assert semantics.clarification_question is None
    assert semantics.cache_scope is CacheScope.CONVERSATION
