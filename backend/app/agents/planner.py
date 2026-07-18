"""Prompt-driven Planner role for the enterprise relationship StateGraph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.prompts import PLANNER_SYSTEM_PROMPT
from app.agents.state import AgentState
from app.llm import ModelClient, ModelInvocationError
from app.schemas import (
    EntityReferenceSource,
    EntityReferenceRole,
    NodeType,
    PlannerDecision,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Planner:
    """Ask the model for entity alignment and an executable research task DAG."""

    model: ModelClient
    max_replans: int = 2
    max_research_steps: int = 8
    entity_catalog: tuple[dict[str, str], ...] = ()
    raw_relation_vocabulary: tuple[str, ...] = ()
    available_tools: tuple[dict[str, str], ...] = ()

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "planner"]
        is_replan = bool(state.get("needs_replan"))
        replan_count = state.get("replan_count", 0) + (1 if is_replan else 0)
        call_updates = {
            "model_call_count": state.get("model_call_count", 0) + 1,
            "planner_model_calls": state.get("planner_model_calls", 0) + 1,
        }
        payload = self._payload(state, replan_count=replan_count)
        try:
            value = await self.model.structured(
                PLANNER_SYSTEM_PROMPT,
                payload,
                PlannerDecision,
                "planner",
            )
            decision = PlannerDecision.model_validate(value)
            self._validate_context_ids(decision, state)
            decision = self._validate_typed_references(decision, state)
            self._validate_research_tasks(decision)
        except ModelInvocationError as exc:
            logger.warning(
                "planner_decision_rejected",
                extra={
                    "event": "planner_decision_rejected",
                    "request_id": state.get("request_id"),
                    "conversation_id": state.get("conversation_id"),
                    "error_type": type(exc).__name__,
                    "is_replan": is_replan,
                },
            )
            return {
                **call_updates,
                "planner_decision": None,
                "planner_failed": True,
                "llm_errors": [*state.get("llm_errors", []), str(exc)],
                "run_status": "failed",
                "needs_replan": False,
                "replan_count": replan_count,
                "route_history": route,
                "agent_steps": [
                    *state.get("agent_steps", []),
                    self._safe_step("provider_failure", "model_invocation_failed"),
                ],
            }
        except (ValidationError, TypeError, ValueError) as exc:
            retry_count = state.get("planner_contract_retry_count", 0)
            first_contract_failure = retry_count == 0
            logger.warning(
                "planner_decision_rejected",
                extra={
                    "event": "planner_decision_rejected",
                    "request_id": state.get("request_id"),
                    "conversation_id": state.get("conversation_id"),
                    "error_type": type(exc).__name__,
                    "is_replan": is_replan,
                    "contract_retry": first_contract_failure,
                },
            )
            update: dict[str, Any] = {
                **call_updates,
                "planner_decision": None,
                "planner_failed": True,
                "run_status": "running" if first_contract_failure else "failed",
                "needs_replan": False,
                "replan_count": replan_count,
                "planner_contract_retry_count": retry_count + 1,
                "route_history": route,
                "agent_steps": [
                    *state.get("agent_steps", []),
                    self._safe_step("contract_rejected", "invalid_typed_contract"),
                ],
            }
            if not first_contract_failure:
                update["llm_errors"] = [
                    *state.get("llm_errors", []),
                    "Planner returned a response that violated its typed contract",
                ]
            return update

        logger.info(
            "planner_decision_accepted",
            extra={
                "event": "planner_decision_accepted",
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id"),
                "intent": decision.intent.value,
                "entity_mention_count": sum(
                    reference.source is EntityReferenceSource.CURRENT_QUERY
                    for reference in decision.entity_references
                ),
                "entity_reference_count": len(decision.entity_references),
                "entity_expected_type_counts": {
                    node_type.value: sum(
                        node_type in reference.expected_types
                        for reference in decision.entity_references
                    )
                    for node_type in NodeType
                },
                "entity_role_counts": {
                    role.value: sum(
                        reference.role is role
                        for reference in decision.entity_references
                    )
                    for role in EntityReferenceRole
                },
                "relation_types": sorted(
                    {
                        relation_type.value
                        for task in decision.research_tasks
                        for relation_type in task.relation_types
                    }
                ),
                "result_merge": decision.result_merge.value,
                "context_entity_count": sum(
                    reference.context_entity_id is not None
                    for reference in decision.entity_references
                ),
                "needs_clarification": decision.intent.value == "clarify",
                "query_requires_realtime_data": decision.query_requires_realtime_data,
                "research_task_count": len(decision.research_tasks),
                "research_task_tool_counts": {
                    tool.value: sum(
                        task.tool is tool for task in decision.research_tasks
                    )
                    for tool in {task.tool for task in decision.research_tasks}
                },
                "canonical_name_count": sum(
                    reference.canonical_name is not None
                    for reference in decision.entity_references
                ),
            },
        )
        resolved = dict(state.get("resolved_entities", {}))
        return {
            **call_updates,
            "planner_decision": decision,
            "planner_failed": False,
            "resolved_entities": resolved,
            "needs_replan": False,
            "replan_count": replan_count,
            "current_replan_reason": None,
            "research_failure_reason": None,
            "research_complete": False,
            "run_status": "running",
            "route_history": route,
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step("plan", None, decision),
            ],
        }

    @staticmethod
    def _safe_step(
        action: str,
        error_code: str | None,
        decision: PlannerDecision | None = None,
    ) -> dict[str, Any]:
        """Return an auditable Planner step without queries, labels, or reasoning."""

        return {
            "role": "planner",
            "action": action,
            "tool": None,
            "relation_types": (
                sorted(
                    {
                        relation_type.value
                        for task in decision.research_tasks
                        for relation_type in task.relation_types
                    }
                )
                if decision is not None
                else []
            ),
            "result_merge": (
                decision.result_merge.value
                if decision is not None
                else None
            ),
            "record_ids": (
                sorted(
                    reference.context_entity_id
                    for reference in decision.entity_references
                    if reference.context_entity_id is not None
                )
                if decision is not None
                else []
            ),
            "argument_fingerprint": None,
            "count": (
                len(decision.entity_references)
                if decision is not None
                else 0
            ),
            "error_code": error_code,
        }

    @staticmethod
    def _validate_context_ids(decision: PlannerDecision, state: AgentState) -> None:
        # A conversational reference may reuse only the explicit begin-turn
        # snapshot of the last successful focus.  Historical aliases, compressed
        # summary entities, and arbitrary nodes still visible in the session graph
        # are not implicit referents for the current sentence.
        allowed_ids = set(
            state.get("prior_focus_entity_ids", state.get("focus_entity_ids", []))
        )
        context_ids = {
            reference.context_entity_id
            for reference in decision.entity_references
            if reference.context_entity_id is not None
        }
        unknown = context_ids - allowed_ids
        if unknown:
            raise ValueError("Planner selected entity IDs outside verified conversation state")
        if (
            any(
                reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
                for reference in decision.entity_references
            )
            and not context_ids
            and decision.intent.value != "clarify"
        ):
            raise ValueError("A context-dependent plan needs verified context IDs or clarification")

    def _validate_typed_references(
        self, decision: PlannerDecision, state: AgentState
    ) -> PlannerDecision:
        """Validate literal mentions and catalog-bound, ID-free new names."""

        query = str(state.get("current_query", ""))
        for reference in decision.entity_references:
            if (
                reference.source is EntityReferenceSource.CURRENT_QUERY
                and reference.mention not in query
            ):
                raise ValueError(
                    "Planner named a current-query entity absent from the query"
                )

        catalog_types: dict[str, set[NodeType]] = {}
        for item in self.entity_catalog:
            try:
                name = str(item["name"])
                node_type = NodeType(str(item["entity_type"]))
            except (KeyError, TypeError, ValueError):
                continue
            catalog_types.setdefault(name, set()).add(node_type)
        for reference in decision.entity_references:
            if reference.source is not EntityReferenceSource.CURRENT_QUERY:
                continue
            canonical_name = reference.canonical_name
            if canonical_name is None:
                continue
            matching_types = catalog_types.get(canonical_name)
            if catalog_types and not matching_types:
                raise ValueError("Planner canonical_name is absent from the raw catalog")
            if matching_types and not matching_types.intersection(
                reference.expected_types
            ):
                raise ValueError(
                    "Planner canonical_name type disagrees with the raw catalog"
                )

        return decision

    def _validate_research_tasks(self, decision: PlannerDecision) -> None:
        """Validate dynamic raw vocabulary without choosing a research step."""

        allowed_raw = set(self.raw_relation_vocabulary)
        for task in decision.research_tasks:
            if allowed_raw and set(task.raw_relation_types) - allowed_raw:
                raise ValueError(
                    "Planner task used a relation word absent from the raw catalog"
                )
            for index in [
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            ]:
                reference = decision.entity_references[index]
                if (
                    task.tool.value == "persons"
                    and NodeType.PERSON not in reference.expected_types
                ):
                    raise ValueError("persons task references a non-person entity")
                if (
                    task.tool.value == "companies"
                    and NodeType.COMPANY not in reference.expected_types
                ):
                    raise ValueError("companies task references a non-company entity")

    @staticmethod
    def _prior_focus_entities(state: AgentState) -> list[dict[str, str]]:
        focus_ids = list(
            state.get("prior_focus_entity_ids", state.get("focus_entity_ids", []))
        )
        if not focus_ids:
            return []
        graph = state.get("session_graph")
        nodes = getattr(graph, "nodes", None)
        if nodes is None and isinstance(graph, dict):
            nodes = graph.get("nodes", [])
        by_id: dict[str, dict[str, str]] = {}
        for node in nodes or []:
            if isinstance(node, dict):
                node_id = node.get("id")
                label = node.get("label")
                node_type = node.get("type")
            else:
                node_id = getattr(node, "id", None)
                label = getattr(node, "label", None)
                node_type = getattr(node, "type", None)
                node_type = getattr(node_type, "value", node_type)
            if node_id and label and node_type:
                by_id[str(node_id)] = {
                    "entity_id": str(node_id),
                    "name": str(label),
                    "entity_type": str(node_type),
                }
        return [by_id[entity_id] for entity_id in focus_ids if entity_id in by_id]

    def _payload(self, state: AgentState, *, replan_count: int) -> dict[str, Any]:
        summary = state.get("summary")
        recent_turns = state.get("recent_turns", [])
        tool_errors = state.get("tool_errors", [])
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "recent_visible_turns": recent_turns,
            "structured_summary": summary,
            "prior_focus_entities": self._prior_focus_entities(state),
            "entity_catalog": list(self.entity_catalog),
            "raw_relation_vocabulary": list(self.raw_relation_vocabulary),
            "available_tools": list(self.available_tools),
            "prior_tool_errors": tool_errors,
            "current_replan_reason": state.get("current_replan_reason"),
            "replan_reason_history": state.get("replan_reasons", []),
            "is_replan": bool(state.get("needs_replan")),
            "planner_contract_retry_count": state.get(
                "planner_contract_retry_count", 0
            ),
            "replan_count": replan_count,
            "limits": {
                "max_replans": self.max_replans,
                "replans_remaining": max(0, self.max_replans - replan_count),
                "max_research_steps": self.max_research_steps,
                "remaining_research_steps": max(
                    0,
                    self.max_research_steps
                    - state.get("research_step_count", 0),
                ),
            },
        }
