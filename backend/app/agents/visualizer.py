"""Prompt-driven Visualizer with deterministic evidence-safe graph projection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.prompts import VISUALIZER_SYSTEM_PROMPT
from app.agents.state import AgentState
from app.state_views import (
    request_semantics,
    signature_focus_entity_ids,
)
from app.evidence_contract import (
    expected_focus_entity_ids,
    requires_explicit_relations,
    validate_signature_records,
)
from app.llm import ModelClient, ModelInvocationError
from app.memory.graph_ops import empty_graph, make_graph
from app.schemas import (
    ControlQueryPolicy,
    Evidence,
    GoalResultStatus,
    GraphEdge,
    GraphNode,
    Intent,
    NodeType,
    QueryGoalSignature,
    QuerySignature,
    RelationType,
    VisualizerDecision,
    VisualizerTextOnlyDecision,
)


ZH_BROAD_CONTROL_DISCLOSURE = (
    "原始数据没有显式控制记录，以下为创办、现任管理或明确持有关系，"
    "不等同法律控制。"
)
EN_BROAD_CONTROL_DISCLOSURE = (
    "The raw data has no explicit control record. The following are founding, "
    "current-management, or explicit ownership associations and are not equivalent "
    "to legal control."
)


logger = logging.getLogger(__name__)


def _goal_query_signature(
    parent: QuerySignature,
    goal: QueryGoalSignature,
) -> QuerySignature:
    """Project one signed goal onto the established evidence validator boundary."""

    return QuerySignature(
        version=parent.version,
        intent=goal.intent,
        subject_ids=goal.subject_ids,
        object_ids=goal.object_ids,
        relation_types=goal.relation_types,
        requested_relation_types=goal.requested_relation_types,
        effective_relation_types=goal.effective_relation_types,
        raw_relation_qualifiers=goal.raw_relation_qualifiers,
        verified_empty_relation_types=goal.verified_empty_relation_types,
        target_types=goal.target_types,
        requested_attributes=goal.requested_attributes,
        context_entity_ids=goal.context_entity_ids,
        result_merge=goal.aggregation,
        control_policy=goal.control_policy,
        control_policy_version=parent.control_policy_version,
        entity_match_version=parent.entity_match_version,
        locale=parent.locale,
    )


def _goal_requires_relations(goal: QueryGoalSignature) -> bool:
    return goal.intent in {
        Intent.FIND_CONTROLLED_COMPANIES,
        Intent.FIND_RELATED_COMPANIES,
        Intent.LOCATE_ENTITIES,
    } or bool(goal.relation_types)


@dataclass(slots=True)
class Visualizer:
    """Ask the model what to show, then materialize only verified tool records."""

    model: ModelClient
    data_version: str

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "visualizer"]
        semantics = request_semantics(state)
        verified_records, evidence = self._verified_catalog(state)
        call_updates = {
            "model_call_count": state.get("model_call_count", 0) + 1,
            "visualizer_model_calls": state.get("visualizer_model_calls", 0) + 1,
        }
        response_model = (
            VisualizerTextOnlyDecision
            if semantics.needs_clarification or state.get("no_match")
            else VisualizerDecision
        )
        try:
            value = await self.model.structured(
                VISUALIZER_SYSTEM_PROMPT,
                self._payload(state, verified_records),
                response_model,
                "visualizer",
            )
            decision = VisualizerDecision.model_validate(value)
            graph = self._project(state, decision, verified_records, evidence)
            normalized_focus_entity_ids = self._normalized_focus(
                state, verified_records, graph
            )
        except (ModelInvocationError, ValidationError, KeyError, TypeError, ValueError) as exc:
            detail = (
                str(exc)
                if isinstance(exc, ModelInvocationError)
                else "Visualizer returned a response that violated the verified-record contract"
            )
            logger.warning(
                "visualizer_decision_rejected",
                extra={
                    "event": "visualizer_decision_rejected",
                    "request_id": state.get("request_id"),
                    "conversation_id": state.get("conversation_id"),
                    "error_type": type(exc).__name__,
                    "reason": (
                        str(exc)
                        if isinstance(exc, (ValidationError, KeyError, TypeError, ValueError))
                        else "model_invocation_error"
                    ),
                },
            )
            return {
                **call_updates,
                **self._contract_rejection(
                    state,
                    route,
                    detail=detail,
                    error_code=(
                        "model_invocation_failed"
                        if isinstance(exc, ModelInvocationError)
                        else "visualizer_contract_rejected"
                    ),
                    retryable=not isinstance(exc, ModelInvocationError),
                ),
            }

        is_clarification = semantics.needs_clarification
        clarification_question = str(semantics.clarification_question or "").strip()
        invalid_clarification = is_clarification and (
            not clarification_question
            or decision.answer.strip() != clarification_question
            or bool(decision.answer_record_ids)
        )
        if invalid_clarification:
            logger.warning(
                "visualizer_clarification_rejected",
                extra={
                    "event": "visualizer_clarification_rejected",
                    "request_id": state.get("request_id"),
                    "conversation_id": state.get("conversation_id"),
                },
            )
            return {
                **call_updates,
                **self._contract_rejection(
                    state,
                    route,
                    detail="Visualizer violated the clarification-only response contract",
                    error_code="clarification_contract_rejected",
                    retryable=True,
                ),
            }

        logger.info(
            "visualizer_selection_accepted",
            extra={
                "event": "visualizer_selection_accepted",
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id"),
                "node_ids": sorted(node.id for node in graph.nodes),
                "edge_ids": sorted(edge.id for edge in graph.edges),
                "focus_entity_ids": normalized_focus_entity_ids,
                "answer_record_ids": sorted(decision.answer_record_ids),
                "is_clarification": is_clarification,
                "no_match": bool(state.get("no_match")),
            },
        )
        return {
            **call_updates,
            "answer": self._with_control_disclosure(
                decision.answer, state, state.get("locale", "zh-CN")
            ),
            "query_result_graph": graph,
            "graph_id": graph.graph_id,
            "focus_entity_ids": (
                list(state.get("focus_entity_ids", []))
                if is_clarification
                else normalized_focus_entity_ids
            ),
            "turn_focus_entity_ids": (
                [] if is_clarification else normalized_focus_entity_ids
            ),
            "agent_steps": [
                *state.get("agent_steps", []),
                {
                    "role": "visualizer",
                    "action": "select_records",
                    "record_ids": sorted(
                        {*(node.id for node in graph.nodes), *(edge.id for edge in graph.edges)}
                    ),
                    "count": len(graph.nodes) + len(graph.edges),
                    "error_code": None,
                },
            ],
            "run_status": "partial" if is_clarification else "success",
            "route_history": route,
        }

    def _project(
        self,
        state: AgentState,
        decision: VisualizerDecision,
        records: list[dict[str, Any]],
        evidence: list[Evidence],
    ):
        entity_records = {
            str(record["id"]): record
            for record in records
            if record.get("record_kind") == "entity" and record.get("id")
        }
        relation_records = {
            str(record["id"]): record
            for record in records
            if record.get("record_kind") == "relation" and record.get("id")
        }
        required = set(state.get("selected_record_ids", []))
        is_clarification = request_semantics(state).needs_clarification
        is_no_match = bool(state.get("no_match"))
        known_record_ids = entity_records.keys() | relation_records.keys()
        if required - known_record_ids:
            raise ValueError("Researcher selected a record absent from verified tools")
        if not is_clarification and not required:
            raise ValueError("a factual result requires Researcher-selected records")
        if is_clarification and required:
            raise ValueError("a clarification cannot carry factual records")
        if set(decision.answer_record_ids) - required:
            raise ValueError("answer_record_ids must come from the Researcher result")
        if required and not decision.answer_record_ids and not is_no_match:
            raise ValueError("a factual answer requires answer_record_ids")

        if not is_clarification and is_no_match:
            signature = QuerySignature.model_validate(state.get("query_signature"))
            self._require_multi_goal_signatures(signature)
            if not self._signature_requires_relations(signature):
                raise ValueError("no_match requires a relational query signature")
            if decision.answer_record_ids:
                raise ValueError("no_match cannot attach entity evidence to an absence claim")
            if required & relation_records.keys():
                raise ValueError("no_match cannot contain relation records")
            if signature.goals and any(
                goal.result_status is GoalResultStatus.NONEMPTY
                for goal in signature.goals
            ):
                raise ValueError("no_match cannot contain a non-empty goal")
            for goal in signature.goals:
                signed_goal_ids = {*goal.subject_ids, *goal.object_ids}
                if goal.result_status is GoalResultStatus.SKIPPED_EMPTY_INPUT:
                    if goal.focus_entity_ids:
                        raise ValueError("a skipped goal cannot retain result focus")
                elif set(goal.focus_entity_ids) != signed_goal_ids:
                    raise ValueError("an empty goal must retain all signed entity focus")
            signed_ids = self._signed_entity_ids(signature)
            if required != signed_ids:
                raise ValueError("no_match nodes must exactly match signed entities")
        elif not is_clarification:
            signature = QuerySignature.model_validate(state.get("query_signature"))
            self._require_multi_goal_signatures(signature)
            selected_records = [
                entity_records[record_id]
                if record_id in entity_records
                else relation_records[record_id]
                for record_id in required
            ]
            if signature.goals:
                self._validate_goal_records(
                    signature,
                    required,
                    selected_records,
                    records,
                )
                self._validate_goal_answer_support(
                    signature,
                    set(decision.answer_record_ids),
                    entity_records,
                    relation_records,
                )
            else:
                validate_signature_records(signature, selected_records, records)
            if not signature.goals and requires_explicit_relations(signature):
                supported_edges = (
                    set(decision.answer_record_ids) & relation_records.keys()
                )
                if not supported_edges:
                    raise ValueError(
                        "a relational answer must be supported by a selected relation record"
                    )

        selected_node_ids = required & entity_records.keys()
        selected_edge_ids = required & relation_records.keys()
        for edge_id in selected_edge_ids:
            record = relation_records[edge_id]
            endpoints = {str(record["source"]), str(record["target"])}
            if not endpoints <= entity_records.keys():
                raise ValueError("selected relation endpoints are absent from verified records")
            # Endpoint closure is a graph invariant, not a new model-selected fact.
            selected_node_ids.update(endpoints)
        nodes = [self._node(entity_records[node_id]) for node_id in selected_node_ids]
        edges = [self._edge(relation_records[edge_id]) for edge_id in selected_edge_ids]
        referenced_evidence_ids = {
            evidence_id
            for element in [*nodes, *edges]
            for evidence_id in element.evidence_ids
        }
        graph_evidence = [
            item for item in evidence if item.id in referenced_evidence_ids
        ]
        return make_graph(nodes, edges, self.data_version, graph_evidence)

    @staticmethod
    def _signature_requires_relations(signature: QuerySignature) -> bool:
        if signature.goals:
            return any(_goal_requires_relations(goal) for goal in signature.goals)
        return requires_explicit_relations(signature)

    @staticmethod
    def _require_multi_goal_signatures(signature: QuerySignature) -> None:
        if signature.intent is Intent.MULTI_GOAL and not signature.goals:
            raise ValueError("multi_goal requires signed per-goal results")

    @staticmethod
    def _signed_entity_ids(signature: QuerySignature) -> set[str]:
        if signature.goals:
            return {
                entity_id
                for goal in signature.goals
                for entity_id in (*goal.subject_ids, *goal.object_ids)
            }
        return {*signature.subject_ids, *signature.object_ids}

    @staticmethod
    def _validate_goal_records(
        signature: QuerySignature,
        required_ids: set[str],
        selected_records: list[dict[str, Any]],
        all_records: list[dict[str, Any]],
    ) -> None:
        records_by_id = {
            str(record["id"]): record
            for record in all_records
            if record.get("id") is not None
        }
        known_ids = set(records_by_id)
        claimed_ids = {
            record_id
            for goal in signature.goals
            for record_id in goal.result_record_ids
        }
        if claimed_ids - required_ids or claimed_ids - known_ids:
            raise ValueError("goal result records must come from the Researcher selection")

        permitted_ids = {
            *claimed_ids,
            *Visualizer._signed_entity_ids(signature),
        }
        for record_id in claimed_ids:
            record = records_by_id[record_id]
            if record.get("record_kind") == "relation":
                permitted_ids.update(
                    str(endpoint)
                    for endpoint in (record.get("source"), record.get("target"))
                    if endpoint is not None
                )
        if required_ids - permitted_ids:
            raise ValueError("Researcher selection contains records outside signed goals")

        selected_ids = {
            str(record["id"])
            for record in selected_records
            if record.get("id") is not None
        }
        for goal in signature.goals:
            signed_ids = {*goal.subject_ids, *goal.object_ids}
            if goal.result_status is GoalResultStatus.SKIPPED_EMPTY_INPUT:
                if goal.focus_entity_ids:
                    raise ValueError("a skipped goal cannot retain result focus")
                continue
            if goal.result_status is GoalResultStatus.VERIFIED_EMPTY:
                if set(goal.focus_entity_ids) != signed_ids:
                    raise ValueError("an empty goal must retain all signed entity focus")
                continue

            goal_ids = set(goal.result_record_ids)
            if not goal_ids or goal_ids - selected_ids:
                raise ValueError("a non-empty goal requires its complete selected records")
            goal_records = [records_by_id[record_id] for record_id in goal_ids]
            goal_signature = _goal_query_signature(signature, goal)
            validate_signature_records(goal_signature, goal_records, all_records)
            expected_focus = expected_focus_entity_ids(
                goal_signature,
                goal_records,
                all_records,
            )
            if set(goal.focus_entity_ids) != set(expected_focus):
                raise ValueError("goal focus does not match its verified records")

    @staticmethod
    def _validate_goal_answer_support(
        signature: QuerySignature,
        answer_record_ids: set[str],
        entity_records: dict[str, dict[str, Any]],
        relation_records: dict[str, dict[str, Any]],
    ) -> None:
        for goal in signature.goals:
            if goal.result_status is not GoalResultStatus.NONEMPTY:
                continue
            answer_kind_ids = (
                relation_records.keys()
                if _goal_requires_relations(goal)
                else entity_records.keys()
            )
            supported = (
                answer_record_ids
                & set(goal.result_record_ids)
                & set(answer_kind_ids)
            )
            if not supported:
                raise ValueError("every non-empty goal requires answer support")

    @staticmethod
    def _normalized_focus(
        state: AgentState,
        records: list[dict[str, Any]],
        graph,
    ) -> list[str]:
        if request_semantics(state).needs_clarification:
            return list(state.get("focus_entity_ids", []))
        signature = QuerySignature.model_validate(state.get("query_signature"))
        goal_focus = signature_focus_entity_ids(signature)
        if state.get("no_match"):
            expected = goal_focus or sorted(
                {*signature.subject_ids, *signature.object_ids}
            )
            graph_node_ids = {node.id for node in graph.nodes}
            if not expected or set(expected) - graph_node_ids:
                raise ValueError(
                    "a verified empty result must retain its signed entity focus"
                )
            return expected
        if goal_focus:
            graph_node_ids = {node.id for node in graph.nodes}
            if set(goal_focus) - graph_node_ids:
                raise ValueError("signed goal focus entities must be selected graph nodes")
            return goal_focus
        required_ids = set(state.get("selected_record_ids", []))
        selected_records = [
            record for record in records if str(record.get("id")) in required_ids
        ]
        expected = expected_focus_entity_ids(signature, selected_records, records)
        graph_node_ids = {node.id for node in graph.nodes}
        if set(expected) - graph_node_ids:
            raise ValueError("normalized focus entities must be selected graph nodes")
        return expected

    @staticmethod
    def _verified_catalog(
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        evidence = [
            item if isinstance(item, Evidence) else Evidence.model_validate(item)
            for item in state.get("tool_evidence", [])
        ]
        evidence_ids = {item.id for item in evidence}
        verified_all: list[dict[str, Any]] = []
        for record in state.get("research_records", []):
            record_evidence = set(record.get("evidence_ids") or [])
            if record_evidence and record_evidence <= evidence_ids:
                verified_all.append(record)

        if request_semantics(state).needs_clarification:
            return [], []
        required_ids = set(state.get("selected_record_ids", []))
        relation_endpoints = {
            str(endpoint)
            for record in verified_all
            if str(record.get("id")) in required_ids
            and record.get("record_kind") == "relation"
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        allowed_ids = required_ids | relation_endpoints
        verified = [
            record
            for record in verified_all
            if str(record.get("id")) in allowed_ids
        ]
        referenced_evidence_ids = {
            evidence_id
            for record in verified
            for evidence_id in record.get("evidence_ids") or []
        }
        return (
            verified,
            [item for item in evidence if item.id in referenced_evidence_ids],
        )

    @staticmethod
    def _node(record: dict[str, Any]) -> GraphNode:
        return GraphNode(
            id=str(record["id"]),
            type=NodeType(record["entity_type"]),
            label=str(record["label"]),
            properties=dict(record.get("properties") or {}),
            evidence_ids=list(record.get("evidence_ids") or []),
        )

    @staticmethod
    def _edge(record: dict[str, Any]) -> GraphEdge:
        return GraphEdge(
            id=str(record["id"]),
            source=str(record["source"]),
            target=str(record["target"]),
            type=RelationType(record["relation_type"]),
            label=str(record["label"]),
            properties=dict(record.get("properties") or {}),
            evidence_ids=list(record.get("evidence_ids") or []),
        )

    def _payload(
        self,
        state: AgentState,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        semantics = request_semantics(state)
        text_only = semantics.needs_clarification or bool(state.get("no_match"))
        graph_record_ids = list(state.get("selected_record_ids", []))
        allowed_answer_record_ids: list[str] = []
        if not text_only:
            signature = QuerySignature.model_validate(state.get("query_signature"))
            graph_ids = set(graph_record_ids)
            if signature.goals:
                allowed_relation_ids = {
                    record_id
                    for goal in signature.goals
                    if goal.result_status is GoalResultStatus.NONEMPTY
                    and _goal_requires_relations(goal)
                    for record_id in goal.result_record_ids
                }
                allowed_entity_ids = {
                    record_id
                    for goal in signature.goals
                    if goal.result_status is GoalResultStatus.NONEMPTY
                    and not _goal_requires_relations(goal)
                    for record_id in goal.result_record_ids
                }
                allowed_answer_record_ids = [
                    str(record["id"])
                    for record in records
                    if str(record.get("id")) in graph_ids
                    and (
                        (
                            record.get("record_kind") == "relation"
                            and str(record.get("id")) in allowed_relation_ids
                        )
                        or (
                            record.get("record_kind") == "entity"
                            and str(record.get("id")) in allowed_entity_ids
                        )
                    )
                ]
            else:
                answer_record_kind = (
                    "relation" if requires_explicit_relations(signature) else "entity"
                )
                allowed_answer_record_ids = [
                    str(record["id"])
                    for record in records
                    if record.get("record_kind") == answer_record_kind
                    and str(record.get("id")) in graph_ids
                ]
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "query_signature": state.get("query_signature"),
            "clarification_question": semantics.clarification_question,
            "no_match": state.get("no_match", False),
            "verified_selected_records": records,
            "graph_record_ids": graph_record_ids,
            "allowed_answer_record_ids": allowed_answer_record_ids,
        }

    @staticmethod
    def _with_control_disclosure(
        answer: str, state: AgentState, locale: str
    ) -> str:
        signature_value = state.get("query_signature")
        if signature_value is not None and not state.get("no_match"):
            signature = QuerySignature.model_validate(signature_value)
            broad_control = (
                signature.intent is Intent.FIND_CONTROLLED_COMPANIES
                and signature.control_policy
                is ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS
                and RelationType.CONTROLS in signature.verified_empty_relation_types
            ) or any(
                goal.intent is Intent.FIND_CONTROLLED_COMPANIES
                and goal.control_policy
                is ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS
                and RelationType.CONTROLS in goal.verified_empty_relation_types
                for goal in signature.goals
            )
            if broad_control:
                control_disclosure = (
                    ZH_BROAD_CONTROL_DISCLOSURE
                    if locale.casefold().startswith("zh")
                    else EN_BROAD_CONTROL_DISCLOSURE
                )
                if control_disclosure not in answer:
                    answer = f"{answer} {control_disclosure}"
        return answer

    def _contract_rejection(
        self,
        state: AgentState,
        route: list[str],
        *,
        detail: str,
        error_code: str,
        retryable: bool,
    ) -> dict[str, Any]:
        graph = empty_graph(self.data_version)
        retry_count = state.get("visualizer_contract_retry_count", 0)
        will_retry = retryable and retry_count < 1
        step = {
            "role": "visualizer",
            "action": "contract_rejected",
            "record_ids": [],
            "count": 0,
            "error_code": error_code,
        }
        return {
            "llm_errors": (
                list(state.get("llm_errors", []))
                if will_retry
                else [*state.get("llm_errors", []), detail]
            ),
            "query_result_graph": graph,
            "graph_id": graph.graph_id,
            "run_status": "running" if will_retry else "failed",
            "visualizer_contract_retry_count": retry_count + int(will_retry),
            "turn_focus_entity_ids": [],
            "agent_steps": [*state.get("agent_steps", []), step],
            "route_history": route,
        }

def error_response(state: AgentState, data_version: str) -> dict[str, Any]:
    """Return a safe generic failure without inventing business facts."""

    graph = empty_graph(data_version)
    locale = state.get("locale", "zh-CN")
    semantics = request_semantics(state)
    is_realtime = semantics.query_requires_realtime_data
    is_unsupported = semantics.intent is Intent.UNSUPPORTED
    if locale.casefold().startswith("zh") and is_realtime:
        answer = "该请求需要实时或外部数据，本地演示工具不支持。"
    elif locale.casefold().startswith("zh") and is_unsupported:
        answer = "该请求超出本地企业关系演示工具的支持范围。"
    elif locale.casefold().startswith("zh"):
        answer = "本次查询未能从本地演示工具生成可验证结果。"
    elif is_realtime:
        answer = "This request needs live or external data that the local demo tools do not support."
    elif is_unsupported:
        answer = "This request is outside the local relationship demo's supported scope."
    else:
        answer = "The local demo tools could not produce a verified result."
    return {
        "answer": answer,
        "query_result_graph": graph,
        "graph_id": graph.graph_id,
        "run_status": "failed",
        "turn_focus_entity_ids": [],
        "research_failure_reason": (
            "realtime_data_unsupported"
            if is_realtime
            else "unsupported_intent"
            if is_unsupported
            else state.get("research_failure_reason")
        ),
        "cache_metadata": state.get("cache_metadata"),
        "route_history": [*state.get("route_history", []), "error_response"],
    }
