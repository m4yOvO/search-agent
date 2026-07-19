"""LangGraph StateGraph assembly, cache paths, and session-memory nodes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agents.planner import Planner
from app.agents.prompts import (
    PLANNER_PROMPT_VERSION,
    RESEARCHER_PROMPT_VERSION,
    VISUALIZER_PROMPT_VERSION,
)
from app.agents.researcher import Researcher, ResultGate, ToolExecutor, route_after_result_gate
from app.agents.state import AgentState, TRANSIENT_DEFAULTS
from app.state_views import request_semantics, signature_focus_entity_ids
from app.agents.visualizer import Visualizer, error_response
from app.config import Settings
from app.llm import ModelClient
from app.memory.chroma_store import CacheLookup, LongTermMemory
from app.memory.compactor import compact_turns
from app.memory.graph_ops import empty_graph, graph_id_for, make_graph, merge_graphs
from app.memory.policy import decide_memory_write
from app.schemas import (
    CacheMetadata,
    CacheScope,
    ConversationSummary,
    ConversationTurn,
    GraphPayload,
    Intent,
    MemoryOperation,
    QuerySignature,
)


PROMPT_VERSIONS = {
    "planner": PLANNER_PROMPT_VERSION,
    "researcher": RESEARCHER_PROMPT_VERSION,
    "visualizer": VISUALIZER_PROMPT_VERSION,
}


def route_after_planner_analysis(state: AgentState) -> str:
    if state.get("planner_failed"):
        if (
            state.get("run_status") == "running"
            and state.get("planner_analysis_retry_count", 0) == 1
        ):
            return "retry"
        return "error"
    if state.get("planner_terminal_review_pending"):
        return "review"
    if state.get("planner_decision") is None:
        return "tasks"
    semantics = request_semantics(state)
    if (
        semantics.query_requires_realtime_data
        or semantics.intent is Intent.UNSUPPORTED
    ):
        return "error"
    if semantics.needs_clarification:
        return "clarify"
    return "error"


def route_after_planner_tasks(state: AgentState) -> str:
    if state.get("planner_failed"):
        if (
            state.get("run_status") == "running"
            and state.get("planner_task_retry_count", 0) == 1
        ):
            return "retry"
        return "error"
    semantics = request_semantics(state)
    if (
        semantics.query_requires_realtime_data
        or semantics.intent is Intent.UNSUPPORTED
    ):
        return "error"
    if semantics.needs_clarification:
        return "clarify"
    return "research"


# Compatibility name for callers that route a fully assembled PlannerDecision.
route_after_planner = route_after_planner_tasks


@dataclass(slots=True, kw_only=True)
class AgentDependencies:
    settings: Settings
    tools: ToolExecutor
    cache: LongTermMemory | None
    data_version: str
    model: ModelClient
    session_graph_validator: Callable[[GraphPayload], bool] | None = None
    planner_catalog: dict[str, Any] = field(default_factory=dict)
    planner_tools: tuple[dict[str, str], ...] = ()


def _cache_update(lookup: CacheLookup) -> dict[str, Any]:
    metadata = CacheMetadata(
        cache_hit=lookup.hit,
        tier="long_term" if lookup.hit else None,
        match_type=lookup.match_type,
        status=lookup.status,
        result_id=lookup.record_id,
        reason=lookup.error,
    )
    return {
        "cache_lookup": lookup,
        "cache_hit": lookup.hit,
        "cache_metadata": metadata,
    }


def cache_focus_entity_ids(payload: Any) -> list[str]:
    """Restore durable focus, preferring signed v5 per-goal focus."""

    goal_focus = signature_focus_entity_ids(payload.query_signature)
    if goal_focus:
        return goal_focus
    if payload.focus_entity_ids:
        return list(payload.focus_entity_ids)
    return list(
        dict.fromkeys(
            [
                *payload.query_signature.context_entity_ids,
                *payload.query_signature.subject_ids,
                *payload.query_signature.object_ids,
            ]
        )
    )


def build_state_graph(deps: AgentDependencies) -> StateGraph:
    research_step_limit = deps.settings.max_research_steps
    hard_iteration_limit = max(
        research_step_limit,
        deps.settings.agent_max_iterations,
    )
    planner = Planner(
        deps.model,
        max_replans=deps.settings.max_replans,
        max_research_steps=hard_iteration_limit,
        input_token_budget=deps.settings.planner_input_token_budget,
        entity_catalog=tuple(deps.planner_catalog.get("entity_catalog", [])),
        raw_relation_vocabulary=tuple(
            deps.planner_catalog.get("raw_relation_vocabulary", [])
        ),
        available_tools=deps.planner_tools,
    )
    researcher = Researcher(
        deps.model,
        deps.tools,
        max_steps=research_step_limit,
        hard_max_steps=hard_iteration_limit,
        max_tool_calls=deps.settings.max_tool_calls,
        max_replans=deps.settings.max_replans,
        retry_step_allowance=deps.settings.research_retry_step_allowance,
        tool_timeout_seconds=deps.settings.tool_timeout_seconds,
        query_signature_version=deps.settings.query_signature_version,
    )
    result_gate = ResultGate(
        max_steps=research_step_limit,
        hard_max_steps=hard_iteration_limit,
        max_replans=deps.settings.max_replans,
        retry_step_allowance=deps.settings.research_retry_step_allowance,
    )
    visualizer = Visualizer(deps.model, deps.data_version)

    def begin_turn(state: AgentState) -> dict[str, Any]:
        reset: dict[str, Any] = copy.deepcopy(TRANSIENT_DEFAULTS)
        recent_turns = [
            item
            if isinstance(item, ConversationTurn)
            else ConversationTurn.model_validate(item)
            for item in state.get("recent_turns", [])
        ]
        summary = (
            state["summary"]
            if isinstance(state.get("summary"), ConversationSummary)
            else ConversationSummary.model_validate(state.get("summary") or {})
        )
        session_value = state.get("session_graph")
        session_invalid = False
        if isinstance(session_value, (GraphPayload, dict)):
            try:
                parsed_session = GraphPayload.model_validate(session_value)
                session_data_version = parsed_session.data_version
                session_invalid = parsed_session.graph_id != graph_id_for(
                    parsed_session.nodes,
                    parsed_session.edges,
                    parsed_session.data_version,
                    parsed_session.evidence,
                )
                if (
                    not session_invalid
                    and deps.session_graph_validator is not None
                    and not deps.session_graph_validator(parsed_session)
                ):
                    session_invalid = True
            except (ValidationError, TypeError, ValueError):
                session_data_version = (
                    session_value.get("data_version")
                    if isinstance(session_value, dict)
                    else session_value.data_version
                )
                session_invalid = True
        else:
            session_data_version = None
            session_invalid = session_value is not None
        prior_data_version = state.get("data_version")
        version_rollover = session_invalid or any(
            value is not None and value != deps.data_version
            for value in (prior_data_version, session_data_version)
        )
        if version_rollover:
            summary = summary.model_copy(
                update={
                    "resolved_entities": {},
                    "focus_entity_ids": [],
                    "confirmed_fact_ids": [],
                    "confirmed_evidence_ids": [],
                    "latest_graph_id": None,
                }
            )

        reset.update(
            {
                "recent_turns": recent_turns,
                "summary": summary,
                "total_turn_count": state.get("total_turn_count", 0),
                "resolved_entities": (
                    {} if version_rollover else dict(state.get("resolved_entities", {}))
                ),
                "focus_entity_ids": (
                    [] if version_rollover else list(state.get("focus_entity_ids", []))
                ),
                "prior_focus_entity_ids": (
                    [] if version_rollover else list(state.get("focus_entity_ids", []))
                ),
                "turn_focus_entity_ids": [],
                "latest_graph_id": (
                    None if version_rollover else state.get("latest_graph_id")
                ),
                "data_version": deps.data_version,
                "model_provider": deps.model.provider,
                "model_name": deps.model.model_name,
                "route_history": ["begin_turn"],
            }
        )
        if version_rollover or not state.get("session_graph"):
            reset["session_graph"] = empty_graph(deps.data_version)
        return reset

    async def raw_cache_probe(state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "raw_cache_probe"]
        if deps.cache is None:
            lookup = CacheLookup(error="cache_unavailable")
        else:
            lookup = await deps.cache.lookup_raw(
                state.get("current_query", ""), state.get("locale", "zh-CN")
            )
        return {**_cache_update(lookup), "route_history": route}

    async def canonical_cache_probe(state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "canonical_cache_probe"]
        signature_value = state.get("query_signature")
        if signature_value is None or deps.cache is None:
            lookup = CacheLookup(error="cache_unavailable")
        else:
            signature = QuerySignature.model_validate(signature_value)
            cache_scope = request_semantics(state).cache_scope
            lookup = await deps.cache.lookup_canonical(
                signature,
                cache_scope=cache_scope,
                conversation_id=(
                    state.get("conversation_id")
                    if cache_scope is CacheScope.CONVERSATION
                    else None
                ),
            )
        return {**_cache_update(lookup), "route_history": route}

    def cache_hydrate(state: AgentState) -> dict[str, Any]:
        lookup = state.get("cache_lookup")
        route = [*state.get("route_history", []), "cache_hydrate"]
        if not lookup or not lookup.hit or not lookup.payload:
            return {
                "run_status": "failed",
                "llm_errors": [
                    *state.get("llm_errors", []),
                    "The selected cache record could not be hydrated",
                ],
                "route_history": route,
            }
        payload = lookup.payload
        hydrated_graph = make_graph(
            payload.graph.nodes,
            payload.graph.edges,
            payload.graph.data_version,
            [*payload.graph.evidence, *payload.evidence],
        )
        # V5 signatures keep the complete follow-up referent set per goal.  It is
        # the durable source on cache hydration; the payload-level list remains a
        # compatibility copy for pre-goal records.
        focus_entity_ids = cache_focus_entity_ids(payload)
        resolved = dict(state.get("resolved_entities", {}))
        resolved.update(payload.resolved_entities)
        return {
            "answer": payload.answer,
            "query_result_graph": hydrated_graph,
            "graph_id": hydrated_graph.graph_id,
            "query_signature": payload.query_signature,
            "focus_entity_ids": focus_entity_ids,
            "turn_focus_entity_ids": focus_entity_ids,
            "query_resolved_entities": dict(payload.resolved_entities),
            "resolved_entities": resolved,
            "run_status": "success",
            "route_history": route,
        }

    async def cache_touch(state: AgentState) -> dict[str, Any]:
        lookup = state.get("cache_lookup")
        route = [*state.get("route_history", []), "cache_touch"]
        if deps.cache is None or lookup is None:
            return {"route_history": route}
        result = await deps.cache.touch(lookup)
        current = state.get("cache_metadata", CacheMetadata())
        metadata = current.model_copy(
            update={
                "status": result.status or current.status,
                "write_operation": result.operation,
                "result_id": result.record_id or current.result_id,
                "reason": result.reason,
            }
        )
        return {"cache_metadata": metadata, "route_history": route}

    async def memory_write(state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "memory_write"]
        decision = decide_memory_write(state)
        current = state.get("cache_metadata", CacheMetadata())
        if decision.operation is not MemoryOperation.ADD:
            return {
                "cache_metadata": current.model_copy(
                    update={"write_operation": decision.operation, "reason": decision.reason}
                ),
                "route_history": route,
            }
        if deps.cache is None:
            reason = "memory_write_failed:cache_unavailable"
            return {
                "cache_metadata": current.model_copy(
                    update={"write_operation": MemoryOperation.SKIP, "reason": reason}
                ),
                "route_history": route,
            }
        signature = QuerySignature.model_validate(state["query_signature"])
        graph = GraphPayload.model_validate(state["query_result_graph"])
        cache_scope = request_semantics(state).cache_scope
        signed_focus = signature_focus_entity_ids(signature)
        result = await deps.cache.write(
            raw_query=state.get("current_query", ""),
            locale=state.get("locale", "zh-CN"),
            signature=signature,
            answer=state.get("answer", ""),
            graph=graph,
            evidence=list(graph.evidence),
            focus_entity_ids=(
                signed_focus
                if signed_focus
                else list(state.get("turn_focus_entity_ids", []))
            ),
            # Cache only aliases established for this query.  Session aliases can
            # include user-specific mentions from unrelated prior turns.
            resolved_entities=dict(state.get("query_resolved_entities", {})),
            cache_scope=cache_scope,
            conversation_id=(
                state.get("conversation_id")
                if cache_scope is CacheScope.CONVERSATION
                else None
            ),
        )
        return {
            "cache_metadata": current.model_copy(
                update={
                    "write_operation": result.operation,
                    "status": result.status,
                    "result_id": result.record_id,
                    "reason": result.reason,
                }
            ),
            "route_history": route,
        }

    def merge_session_graph(state: AgentState) -> dict[str, Any]:
        previous_value = state.get("session_graph")
        previous = GraphPayload.model_validate(previous_value) if previous_value else None
        delta_value = state.get("query_result_graph")
        delta = (
            GraphPayload.model_validate(delta_value)
            if delta_value
            else empty_graph(deps.data_version)
        )
        merged = merge_graphs(previous, delta)
        return {
            "session_graph": merged,
            "graph_id": merged.graph_id,
            "latest_graph_id": merged.graph_id,
            "route_history": [*state.get("route_history", []), "merge_session_graph"],
        }

    def compact_session(state: AgentState) -> dict[str, Any]:
        turns = [
            item if isinstance(item, ConversationTurn) else ConversationTurn.model_validate(item)
            for item in state.get("recent_turns", [])
        ]
        turns.append(
            ConversationTurn(
                user=state.get("current_query", ""),
                assistant=state.get("answer", ""),
                intent=request_semantics(state).intent,
                focus_entity_ids=list(state.get("turn_focus_entity_ids", [])),
            )
        )
        summary_value = state.get("summary")
        summary = (
            summary_value
            if isinstance(summary_value, ConversationSummary)
            else ConversationSummary.model_validate(summary_value or {})
        )
        query_graph_value = state.get("query_result_graph")
        query_graph = (
            GraphPayload.model_validate(query_graph_value)
            if query_graph_value
            else empty_graph(deps.data_version)
        )
        fact_ids = [
            *(node.id for node in query_graph.nodes),
            *(edge.id for edge in query_graph.edges),
        ]
        retained, summary = compact_turns(
            turns,
            summary,
            max_turns=deps.settings.short_term_max_turns,
            compact_oldest=deps.settings.short_term_compact_oldest,
            keep_recent=deps.settings.short_term_keep_recent,
            resolved_entities=state.get("resolved_entities", {}),
            focus_entity_ids=state.get("focus_entity_ids", []),
            fact_ids=fact_ids,
            evidence=query_graph.evidence,
            latest_graph_id=state.get("graph_id"),
        )
        return {
            "recent_turns": retained,
            "summary": summary,
            "total_turn_count": state.get("total_turn_count", 0) + 1,
            # Raw tool payloads are per-request scratch space and are not retained in
            # the checkpoint after their verified graph has been materialized.
            "research_records": [],
            "tool_evidence": [],
            "research_transcript": [],
            "route_history": [*state.get("route_history", []), "compact_session"],
        }

    builder = StateGraph(AgentState)
    builder.add_node("begin_turn", begin_turn)
    builder.add_node("raw_cache_probe", raw_cache_probe)
    builder.add_node("planner_analyze", planner.analyze)
    builder.add_node("planner_tasks", planner.plan_tasks)
    builder.add_node("researcher", researcher)
    builder.add_node("result_gate", result_gate)
    builder.add_node("canonical_cache_probe", canonical_cache_probe)
    builder.add_node("cache_hydrate", cache_hydrate)
    builder.add_node("cache_touch", cache_touch)
    builder.add_node("visualizer", visualizer)
    builder.add_node("memory_write", memory_write)
    builder.add_node("merge_session_graph", merge_session_graph)
    builder.add_node("compact_session", compact_session)
    builder.add_node("error_response", lambda state: error_response(state, deps.data_version))

    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "raw_cache_probe")
    builder.add_conditional_edges(
        "raw_cache_probe",
        lambda state: "hit" if state.get("cache_hit") else "miss",
        {"hit": "cache_hydrate", "miss": "planner_analyze"},
    )
    builder.add_conditional_edges(
        "planner_analyze",
        route_after_planner_analysis,
        {
            "retry": "planner_analyze",
            "review": "planner_analyze",
            "error": "error_response",
            "clarify": "visualizer",
            "tasks": "planner_tasks",
        },
    )
    builder.add_conditional_edges(
        "planner_tasks",
        route_after_planner_tasks,
        {
            "retry": "planner_tasks",
            "error": "error_response",
            "clarify": "visualizer",
            "research": "researcher",
        },
    )
    builder.add_edge("researcher", "result_gate")
    builder.add_conditional_edges(
        "result_gate",
        route_after_result_gate,
        {
            "research": "researcher",
            "replan": "planner_analyze",
            "no_match": "visualizer",
            "valid": "canonical_cache_probe",
            "error": "error_response",
        },
    )
    builder.add_conditional_edges(
        "canonical_cache_probe",
        lambda state: "hit" if state.get("cache_hit") else "miss",
        {"hit": "cache_hydrate", "miss": "visualizer"},
    )
    builder.add_edge("cache_hydrate", "cache_touch")
    builder.add_edge("cache_touch", "merge_session_graph")
    builder.add_conditional_edges(
        "visualizer",
        lambda state: (
            "retry"
            if state.get("run_status") == "running"
            and state.get("visualizer_contract_retry_count", 0) == 1
            else "error"
            if state.get("run_status") == "failed"
            else "complete"
        ),
        {
            "retry": "visualizer",
            "error": "error_response",
            "complete": "memory_write",
        },
    )
    builder.add_edge("memory_write", "merge_session_graph")
    builder.add_edge("error_response", "merge_session_graph")
    builder.add_edge("merge_session_graph", "compact_session")
    builder.add_edge("compact_session", END)
    return builder


def compile_agent_graph(deps: AgentDependencies, checkpointer: Any | None = None) -> Any:
    return build_state_graph(deps).compile(checkpointer=checkpointer)


build_agent_graph = build_state_graph
