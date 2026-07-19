"""Deterministic long-term memory write gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.evidence_contract import requires_explicit_relations
from app.memory.graph_ops import evidence_coverage, graph_is_valid
from app.schemas import (
    Evidence,
    GoalResultStatus,
    GraphPayload,
    Intent,
    MemoryOperation,
    QuerySignature,
)
from app.state_views import request_semantics


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    operation: MemoryOperation
    reason: str


def decide_memory_write(state: Mapping[str, Any]) -> MemoryDecision:
    if state.get("cache_hit"):
        return MemoryDecision(MemoryOperation.SKIP, "cache_hit_uses_touch_path")
    if state.get("no_match"):
        return MemoryDecision(MemoryOperation.SKIP, "verified_empty_result")
    if state.get("run_status") != "success":
        return MemoryDecision(MemoryOperation.SKIP, "run_not_successful")
    if not state.get("research_complete"):
        return MemoryDecision(MemoryOperation.SKIP, "research_incomplete")
    if state.get("tool_call_count", 0) < 1 or not state.get("selected_record_ids"):
        return MemoryDecision(MemoryOperation.SKIP, "missing_verified_tool_selection")
    if state.get("tool_errors"):
        return MemoryDecision(MemoryOperation.SKIP, "tool_error")
    if state.get("llm_errors"):
        return MemoryDecision(MemoryOperation.SKIP, "model_error")
    semantics = request_semantics(state)
    if semantics.needs_clarification:
        return MemoryDecision(MemoryOperation.SKIP, "ambiguous_or_clarification")
    if semantics.query_requires_realtime_data:
        return MemoryDecision(MemoryOperation.SKIP, "realtime_query")

    intent = semantics.intent
    if intent is None:
        return MemoryDecision(MemoryOperation.SKIP, "invalid_intent")
    if intent in {Intent.CLARIFY, Intent.UNSUPPORTED}:
        return MemoryDecision(MemoryOperation.SKIP, "non_cacheable_intent")

    graph_value = state.get("query_result_graph")
    if not graph_is_valid(graph_value):
        return MemoryDecision(MemoryOperation.SKIP, "invalid_graph")
    graph = GraphPayload.model_validate(graph_value)
    if not graph.nodes:
        return MemoryDecision(MemoryOperation.SKIP, "empty_result")
    evidence = [Evidence.model_validate(item) for item in state.get("tool_evidence", [])]
    if evidence_coverage(graph, evidence) < 1.0:
        return MemoryDecision(MemoryOperation.SKIP, "incomplete_evidence")
    if not state.get("query_signature") or not state.get("answer"):
        return MemoryDecision(MemoryOperation.SKIP, "missing_cache_payload")
    signature = QuerySignature.model_validate(state["query_signature"])
    if signature.intent is Intent.MULTI_GOAL and not signature.goals:
        return MemoryDecision(MemoryOperation.SKIP, "missing_goal_signatures")
    if signature.goals:
        graph_record_ids = {
            *(node.id for node in graph.nodes),
            *(edge.id for edge in graph.edges),
        }
        graph_node_ids = {node.id for node in graph.nodes}
        graph_edge_ids = {edge.id for edge in graph.edges}
        signed_focus = {
            entity_id
            for goal in signature.goals
            for entity_id in goal.focus_entity_ids
        }
        if signed_focus != set(state.get("turn_focus_entity_ids", [])):
            return MemoryDecision(MemoryOperation.SKIP, "goal_focus_mismatch")
        if signed_focus - graph_node_ids:
            return MemoryDecision(MemoryOperation.SKIP, "goal_focus_outside_graph")
        for goal in signature.goals:
            result_ids = set(goal.result_record_ids)
            if result_ids - graph_record_ids:
                return MemoryDecision(
                    MemoryOperation.SKIP,
                    "goal_records_outside_graph",
                )
            if goal.result_status is not GoalResultStatus.NONEMPTY:
                continue
            relation_goal = goal.intent in {
                Intent.FIND_CONTROLLED_COMPANIES,
                Intent.FIND_RELATED_COMPANIES,
                Intent.LOCATE_ENTITIES,
            } or bool(goal.relation_types)
            if relation_goal and not (result_ids & graph_edge_ids):
                return MemoryDecision(
                    MemoryOperation.SKIP,
                    "missing_required_relation_evidence",
                )
    elif requires_explicit_relations(signature) and not graph.edges:
        return MemoryDecision(MemoryOperation.SKIP, "missing_required_relation_evidence")
    return MemoryDecision(MemoryOperation.ADD, "first_verified_result")
