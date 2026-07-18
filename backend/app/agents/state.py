"""Shared LangGraph state for the prompt-driven role agents."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.channels import UntrackedValue

from app.memory.chroma_store import CacheLookup
from app.schemas import (
    CacheMetadata,
    ConversationSummary,
    ConversationTurn,
    Evidence,
    GraphPayload,
    PlannerDecision,
    QuerySignature,
    ToolError,
)


class AgentState(TypedDict, total=False):
    # Bounded conversation state persisted by the LangGraph checkpointer.
    conversation_id: str
    recent_turns: list[ConversationTurn]
    summary: ConversationSummary
    total_turn_count: int
    resolved_entities: dict[str, str]
    focus_entity_ids: list[str]
    latest_graph_id: str | None
    data_version: str
    session_graph: GraphPayload

    # Per-request input. These values remain available for the live graph run but
    # are deliberately absent when a checkpoint is restored.
    request_id: Annotated[str, UntrackedValue]
    current_query: Annotated[str, UntrackedValue]
    locale: Annotated[str, UntrackedValue]
    # Snapshot of the last successful persisted focus at begin_turn.  This is
    # Planner input only; it is never treated as the current query's result.
    prior_focus_entity_ids: Annotated[list[str], UntrackedValue]
    # Verified referents produced by this turn.  Failure and clarification keep
    # this empty even while persisted ``focus_entity_ids`` remains available.
    turn_focus_entity_ids: Annotated[list[str], UntrackedValue]

    # Planner request state.
    planner_decision: Annotated[PlannerDecision | None, UntrackedValue]
    planner_failed: Annotated[bool, UntrackedValue]
    query_signature: Annotated[QuerySignature | None, UntrackedValue]
    needs_replan: Annotated[bool, UntrackedValue]
    replan_count: Annotated[int, UntrackedValue]
    current_replan_reason: Annotated[str | None, UntrackedValue]
    replan_reasons: Annotated[list[str], UntrackedValue]
    research_failure_reason: Annotated[str | None, UntrackedValue]
    planner_contract_retry_count: Annotated[int, UntrackedValue]
    researcher_contract_retry_count: Annotated[int, UntrackedValue]
    visualizer_contract_retry_count: Annotated[int, UntrackedValue]

    # Researcher request state, including raw mock-tool records and transcript.
    research_records: Annotated[list[dict[str, Any]], UntrackedValue]
    selected_record_ids: Annotated[list[str], UntrackedValue]
    # Aliases evidenced for this query only.  The session-level
    # ``resolved_entities`` map may contain private mentions from prior turns and
    # must never be serialized into a public long-term cache record wholesale.
    query_resolved_entities: Annotated[dict[str, str], UntrackedValue]
    research_transcript: Annotated[list[dict[str, Any]], UntrackedValue]
    research_complete: Annotated[bool, UntrackedValue]
    # True only when an exhaustive, correctly scoped mock relation call succeeded
    # with zero records. This is a verified empty result, not a tool/model failure.
    no_match: Annotated[bool, UntrackedValue]
    tool_evidence: Annotated[list[Evidence], UntrackedValue]
    tool_errors: Annotated[list[ToolError], UntrackedValue]
    research_step_count: Annotated[int, UntrackedValue]
    tool_call_count: Annotated[int, UntrackedValue]
    researcher_invoked: Annotated[bool, UntrackedValue]
    executed_tool_fingerprints: Annotated[dict[str, str], UntrackedValue]
    run_status: Annotated[
        Literal["running", "success", "partial", "failed"], UntrackedValue
    ]

    # Per-request model trace and safe failures.
    model_provider: Annotated[str, UntrackedValue]
    model_name: Annotated[str, UntrackedValue]
    model_call_count: Annotated[int, UntrackedValue]
    planner_model_calls: Annotated[int, UntrackedValue]
    researcher_model_calls: Annotated[int, UntrackedValue]
    visualizer_model_calls: Annotated[int, UntrackedValue]
    llm_errors: Annotated[list[str], UntrackedValue]
    agent_steps: Annotated[list[dict[str, Any]], UntrackedValue]

    # Per-request cache state.
    cache_lookup: Annotated[CacheLookup | None, UntrackedValue]
    cache_hit: Annotated[bool, UntrackedValue]
    cache_metadata: Annotated[CacheMetadata, UntrackedValue]

    # Per-request output and route trace. The user-visible answer is retained only
    # through the bounded ConversationTurn history above.
    answer: Annotated[str, UntrackedValue]
    query_result_graph: Annotated[GraphPayload, UntrackedValue]
    graph_id: Annotated[str, UntrackedValue]
    route_history: Annotated[list[str], UntrackedValue]


TRANSIENT_DEFAULTS: dict[str, Any] = {
    "prior_focus_entity_ids": [],
    "turn_focus_entity_ids": [],
    "planner_decision": None,
    "planner_failed": False,
    "query_signature": None,
    "needs_replan": False,
    "replan_count": 0,
    "current_replan_reason": None,
    "replan_reasons": [],
    "research_failure_reason": None,
    "planner_contract_retry_count": 0,
    "researcher_contract_retry_count": 0,
    "visualizer_contract_retry_count": 0,
    "research_records": [],
    "selected_record_ids": [],
    "query_resolved_entities": {},
    "research_transcript": [],
    "research_complete": False,
    "no_match": False,
    "tool_evidence": [],
    "tool_errors": [],
    "research_step_count": 0,
    "tool_call_count": 0,
    "researcher_invoked": False,
    "executed_tool_fingerprints": {},
    "run_status": "running",
    "model_call_count": 0,
    "planner_model_calls": 0,
    "researcher_model_calls": 0,
    "visualizer_model_calls": 0,
    "llm_errors": [],
    "agent_steps": [],
    "cache_lookup": None,
    "cache_hit": False,
    "cache_metadata": CacheMetadata(),
    "answer": "",
    "query_result_graph": None,
    "graph_id": "",
    "route_history": [],
}
