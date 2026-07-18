"""Task-DAG driven Researcher and deterministic evidence guardrails.

Planner owns intent, entity alignment, and task decomposition.  This module never
re-interprets the user's words.  It exposes only currently executable fact tools,
validates the model-selected call against the ready Planner task(s), records the
typed receipt, and derives completion from those receipts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.prompts import RESEARCHER_SYSTEM_PROMPT
from app.agents.state import AgentState
from app.evidence_contract import expected_focus_entity_ids, validate_signature_records
from app.llm import ModelClient, ModelInvocationError, NativeToolCall
from app.schemas import (
    ControlQueryPolicy,
    EntityReferenceSource,
    Evidence,
    Intent,
    NodeType,
    PlannerDecision,
    QuerySignature,
    RelationType,
    ResearchAction,
    ResearchDirection,
    ResearchTask,
    ResearcherDecision,
    ResultMergeStrategy,
    ToolError,
    ToolName,
    ToolResult,
)
from app.tools.contracts import (
    ENTITY_MATCH_ALGORITHM_VERSION,
    CompaniesRequest,
    MatchMode,
    PersonsRequest,
    RelationDirection,
    RelationsRequest,
    ToolResultMeta,
    TypedToolResult,
)
from app.tools.specs import make_openai_strict_schema


logger = logging.getLogger(__name__)


class ToolExecutor(Protocol):
    async def execute(self, tool: ToolName | str, arguments: dict[str, Any]) -> ToolResult: ...


class _CompletionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FailureArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    failure_message: str = Field(min_length=1, max_length=500)


_LIFECYCLE_MODELS: dict[str, type[BaseModel]] = {
    ResearchAction.FINISH.value: _CompletionArguments,
    ResearchAction.NO_RESULTS.value: _CompletionArguments,
    ResearchAction.REPLAN.value: _FailureArguments,
    ResearchAction.FAIL.value: _FailureArguments,
}

_LIFECYCLE_DESCRIPTIONS = {
    ResearchAction.FINISH.value: (
        "所有 Planner 任务已有完整成功回执且合并结果非空时发出完成信号；无需参数。"
    ),
    ResearchAction.NO_RESULTS.value: (
        "所有 Planner 任务已有完整成功回执且最终合并结果为空时发出信号；无需参数。"
    ),
    ResearchAction.REPLAN.value: "任务 DAG 无法按当前计划完成时，向 Planner 请求有界重规划。",
    ResearchAction.FAIL.value: "限制耗尽或遇到不可恢复条件时安全终止。",
}

_TOOL_REQUEST_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.PERSONS: PersonsRequest,
    ToolName.COMPANIES: CompaniesRequest,
    ToolName.RELATIONS: RelationsRequest,
}

_RELATION_FIELDS = (
    "subject_ids",
    "object_ids",
    "relation_types",
    "raw_relation_types",
    "direction",
    "include_endpoints",
    "limit",
)


def _effective_research_step_limit(
    state: AgentState,
    *,
    base_steps: int,
    hard_max_steps: int,
    max_replans: int,
    retry_step_allowance: int,
) -> int:
    replan_bonus = (
        min(max(0, int(state.get("replan_count", 0))), max_replans)
        * retry_step_allowance
    )
    contract_bonus = min(
        max(0, int(state.get("researcher_contract_retry_count", 0))), 1
    )
    return min(hard_max_steps, base_steps + replan_bonus + contract_bonus)


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("record_kind", "")), str(record.get("id", "")))
        if key[1]:
            unique[key] = record
    return list(unique.values())


def _deduplicate_evidence(evidence: list[Evidence]) -> list[Evidence]:
    result: dict[str, Evidence] = {}
    for item in evidence:
        previous = result.get(item.id)
        if previous is None or item.retrieved_at > previous.retrieved_at:
            result[item.id] = item
    return sorted(result.values(), key=lambda item: item.id)


@dataclass(frozen=True, slots=True)
class _TaskContract:
    task_ids: tuple[str, ...]
    tool: ToolName
    reference_indexes: tuple[int, ...]
    arguments: dict[str, Any] | None = None
    candidate_queries: tuple[str, ...] = ()
    allowed_entity_ids: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()

    def public_summary(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "task_ids": list(self.task_ids),
            "tool": self.tool.value,
            "reference_indexes": list(self.reference_indexes),
        }
        if self.arguments is not None:
            value["required_arguments"] = self.arguments
        else:
            value["candidate_queries"] = list(self.candidate_queries)
            value["allowed_entity_ids"] = list(self.allowed_entity_ids)
            value["requested_attributes"] = list(self.requested_attributes)
        return value


@dataclass(slots=True)
class _TaskSnapshot:
    planner: PlannerDecision
    bindings: dict[int, str]
    task_receipts: dict[str, list[dict[str, Any]]]
    completed: set[str]
    contracts: list[_TaskContract]
    terminal_task_ids: set[str]
    selected_records: list[dict[str, Any]]
    result_nonempty: bool

    @property
    def all_complete(self) -> bool:
        return bool(self.planner.research_tasks) and self.completed == {
            task.task_id for task in self.planner.research_tasks
        }


@dataclass(slots=True)
class Researcher:
    """Execute one model-selected action against the Planner task DAG per turn."""

    model: ModelClient
    tools: ToolExecutor
    max_steps: int = 12
    hard_max_steps: int | None = None
    max_tool_calls: int = 10
    max_replans: int = 2
    retry_step_allowance: int = 3
    tool_timeout_seconds: float = 5.0
    query_signature_version: int = 4

    def _hard_step_limit(self) -> int:
        return max(self.max_steps, self.hard_max_steps or self.max_steps)

    def _effective_step_limit(self, state: AgentState) -> int:
        return _effective_research_step_limit(
            state,
            base_steps=self.max_steps,
            hard_max_steps=self._hard_step_limit(),
            max_replans=self.max_replans,
            retry_step_allowance=self.retry_step_allowance,
        )

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "researcher"]
        steps = int(state.get("research_step_count", 0))
        if steps >= self._hard_step_limit() or steps >= self._effective_step_limit(state):
            return self._terminal_failure(
                state,
                "Researcher exhausted its bounded model-step budget",
                route,
            )

        counters = {
            "research_step_count": steps + 1,
            "model_call_count": int(state.get("model_call_count", 0)) + 1,
            "researcher_model_calls": int(state.get("researcher_model_calls", 0)) + 1,
            "researcher_invoked": True,
        }
        try:
            value = await self._invoke_model(state)
            decision = ResearcherDecision.model_validate(value)
        except ModelInvocationError as exc:
            logger.warning(
                "researcher_provider_failure",
                extra=self._log_context(state, steps + 1, "provider_failure"),
            )
            return {
                **counters,
                "llm_errors": [*state.get("llm_errors", []), str(exc)],
                "run_status": "failed",
                "research_complete": False,
                "route_history": route,
                "agent_steps": [
                    *state.get("agent_steps", []),
                    self._safe_step("provider_failure", error_code="model_invocation_failed"),
                ],
            }
        except (ValidationError, TypeError, ValueError):
            return self._contract_failure(
                state,
                counters,
                route,
                error_code="invalid_typed_contract",
            )

        base = {
            **counters,
            "route_history": route,
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step(
                    decision.action.value,
                    tool=decision.tool,
                    record_ids=decision.selected_record_ids,
                ),
            ],
        }
        logger.info(
            "researcher_action",
            extra={
                **self._log_context(state, steps + 1, decision.action.value),
                "tool": decision.tool.value if decision.tool else None,
            },
        )

        if decision.action is ResearchAction.CALL_TOOL:
            return {**base, **(await self._execute_tool_call(state, decision))}
        if decision.action in {ResearchAction.FINISH, ResearchAction.NO_RESULTS}:
            try:
                return {**base, **self._finish(state, decision)}
            except (ValidationError, TypeError, ValueError):
                return self._completion_failure(state, base)
        if decision.action is ResearchAction.REPLAN:
            reason = decision.failure_message or "Researcher requested replanning"
            return {
                **base,
                "needs_replan": True,
                "current_replan_reason": reason,
                "replan_reasons": [*state.get("replan_reasons", []), reason],
                "research_complete": False,
                "run_status": "running",
            }
        reason = decision.failure_message or "Researcher could not complete the plan"
        return {
            **base,
            "run_status": "failed",
            "research_complete": False,
            "research_failure_reason": reason,
        }

    async def _invoke_model(self, state: AgentState) -> ResearcherDecision | dict[str, Any]:
        payload = self._payload(state)
        native_call = getattr(self.model, "researcher_tool_call", None)
        if callable(native_call):
            schema_factory = getattr(self.tools, "openai_function_schemas", None)
            if not callable(schema_factory):
                raise ValueError("Researcher tools must expose native function schemas")
            definitions = [
                *self._available_fact_tool_definitions(
                    state, list(schema_factory())
                ),
                *self._available_lifecycle_tool_definitions(state),
            ]
            if not definitions:
                raise ValueError("no valid Researcher action is available")
            call = NativeToolCall.model_validate(
                await native_call(
                    RESEARCHER_SYSTEM_PROMPT,
                    payload,
                    definitions,
                    "researcher",
                )
            )
            return self._decision_from_native_call(call)
        return await self.model.structured(
            RESEARCHER_SYSTEM_PROMPT,
            payload,
            ResearcherDecision,
            "researcher",
        )

    @staticmethod
    def _decision_from_native_call(call: NativeToolCall) -> ResearcherDecision:
        if call.name in {tool.value for tool in ToolName}:
            return ResearcherDecision(
                action=ResearchAction.CALL_TOOL,
                tool=ToolName(call.name),
                arguments=call.arguments,
            )
        try:
            action = ResearchAction(call.name)
            model = _LIFECYCLE_MODELS[call.name]
        except (KeyError, ValueError) as exc:
            raise ValueError("unknown Researcher lifecycle action") from exc
        arguments = model.model_validate(call.arguments)
        return ResearcherDecision(
            action=action,
            failure_message=(
                getattr(arguments, "failure_message", None)
                if action in {ResearchAction.REPLAN, ResearchAction.FAIL}
                else None
            ),
        )

    def _available_fact_tool_definitions(
        self,
        state: AgentState,
        definitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        snapshot = self._snapshot(state)
        if snapshot.all_complete or int(state.get("tool_call_count", 0)) >= self.max_tool_calls:
            return []
        by_name = {
            str(definition.get("name")): copy.deepcopy(definition)
            for definition in definitions
        }
        result: list[dict[str, Any]] = []
        for tool in ToolName:
            contracts = [item for item in snapshot.contracts if item.tool is tool]
            definition = by_name.get(tool.value)
            if not contracts or definition is None:
                continue
            if tool is ToolName.RELATIONS and len(contracts) == 1:
                assert contracts[0].arguments is not None
                self._scope_relation_schema(definition, contracts[0].arguments)
            result.append(definition)
        return result

    def _available_lifecycle_tool_definitions(
        self, state: AgentState
    ) -> list[dict[str, Any]]:
        snapshot = self._snapshot(state)
        if snapshot.all_complete:
            action = (
                ResearchAction.FINISH
                if snapshot.result_nonempty
                else ResearchAction.NO_RESULTS
            )
            return [self._lifecycle_definition(action)]
        # While a Planner task is executable the model should act on that task,
        # not abandon a valid DAG path.  Replan/fail become available only when
        # the verified state leaves no executable contract.
        if snapshot.contracts:
            return []
        return [
            self._lifecycle_definition(ResearchAction.REPLAN),
            self._lifecycle_definition(ResearchAction.FAIL),
        ]

    @staticmethod
    def _lifecycle_definition(action: ResearchAction) -> dict[str, Any]:
        model = _LIFECYCLE_MODELS[action.value]
        parameters = copy.deepcopy(model.model_json_schema())
        make_openai_strict_schema(parameters)
        return {
            "type": "function",
            "name": action.value,
            "description": _LIFECYCLE_DESCRIPTIONS[action.value],
            "parameters": parameters,
            "strict": True,
        }

    @staticmethod
    def _scope_relation_schema(
        definition: dict[str, Any], arguments: dict[str, Any]
    ) -> None:
        parameters = definition.get("parameters", {})
        properties = parameters.get("properties", {})
        parameters["required"] = list(_RELATION_FIELDS)
        for field in _RELATION_FIELDS:
            schema = properties.get(field)
            if not isinstance(schema, dict):
                continue
            value = arguments[field]
            if isinstance(value, list):
                schema["minItems"] = len(value)
                schema["maxItems"] = len(value)
                item_schema = schema.setdefault("items", {})
                if value:
                    item_schema["enum"] = list(value)
                else:
                    item_schema.pop("enum", None)
            else:
                schema["enum"] = [value]

    async def _execute_tool_call(
        self, state: AgentState, decision: ResearcherDecision
    ) -> dict[str, Any]:
        assert decision.tool is not None
        try:
            request_model = _TOOL_REQUEST_MODELS[decision.tool]
            request = request_model.model_validate(decision.arguments)
            contracts = self._matching_contracts(
                state, decision.tool, decision.arguments, request
            )
            if not contracts:
                raise ValueError("tool call does not match a ready Planner task")
            self._validate_trusted_ids(state, decision.tool, request)
        except (KeyError, ValidationError, TypeError, ValueError):
            return self._tool_contract_failure(
                state,
                decision,
                "task_scope_mismatch",
            )

        dumped_arguments = request.model_dump(mode="json")
        normalized_arguments = (
            {
                field: (
                    sorted(dumped_arguments[field])
                    if field
                    in {
                        "subject_ids",
                        "object_ids",
                        "relation_types",
                        "raw_relation_types",
                    }
                    else dumped_arguments[field]
                )
                for field in _RELATION_FIELDS
            }
            if isinstance(request, RelationsRequest)
            else dumped_arguments
        )
        fingerprint = self._tool_call_fingerprint(decision.tool, normalized_arguments)
        fingerprints = dict(state.get("executed_tool_fingerprints", {}))
        if fingerprint in fingerprints:
            reason = "Researcher proposed a duplicate deterministic tool call"
            return {
                "research_transcript": [
                    *state.get("research_transcript", []),
                    self._receipt(
                        state,
                        decision.tool,
                        normalized_arguments,
                        contracts,
                        executed=False,
                        success=False,
                        error_code="duplicate_tool_call",
                        fingerprint=fingerprint,
                    ),
                ],
                "needs_replan": True,
                "current_replan_reason": reason,
                "replan_reasons": [*state.get("replan_reasons", []), reason],
                "research_complete": False,
                "run_status": "running",
                "agent_steps": [
                    *state.get("agent_steps", []),
                    self._safe_step(
                        "duplicate_tool_call",
                        tool=decision.tool,
                        fingerprint=fingerprint,
                        error_code="duplicate_tool_call",
                    ),
                ],
            }

        if int(state.get("tool_call_count", 0)) >= self.max_tool_calls:
            return self._terminal_failure(
                state,
                "Researcher exhausted its fact-tool call budget",
                list(state.get("route_history", [])),
            )

        try:
            raw_result = await asyncio.wait_for(
                self.tools.execute(decision.tool, normalized_arguments),
                timeout=self.tool_timeout_seconds,
            )
            result = TypedToolResult.model_validate(
                raw_result.model_dump(mode="python")
                if isinstance(raw_result, BaseModel)
                else raw_result
            )
        except TimeoutError:
            result = self._synthetic_tool_failure(
                state,
                decision.tool,
                "tool_timeout",
                "Local mock tool timed out",
                retryable=True,
            )
        except (ValidationError, TypeError, ValueError):
            result = self._synthetic_tool_failure(
                state,
                decision.tool,
                "invalid_tool_result",
                "Local mock tool returned an invalid typed result",
            )

        fingerprints[fingerprint] = (
            "success" if result.success else (result.error.code if result.error else "error")
        )
        record_ids = [
            str(record.get("id"))
            for record in result.records
            if record.get("id") is not None
        ]
        receipt = self._receipt(
            state,
            decision.tool,
            normalized_arguments,
            contracts,
            executed=True,
            success=result.success,
            error_code=result.error.code if result.error else None,
            fingerprint=fingerprint,
            record_ids=record_ids,
            meta=result.meta.model_dump(mode="json"),
            data_version=result.data_version,
        )
        transcript = [*state.get("research_transcript", []), receipt]
        if not result.success:
            errors = list(state.get("tool_errors", []))
            if result.error is not None:
                errors.append(result.error)
            return {
                "research_transcript": transcript,
                "tool_errors": errors,
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
                "executed_tool_fingerprints": fingerprints,
                "research_complete": False,
                "run_status": "running",
                "agent_steps": [
                    *state.get("agent_steps", []),
                    self._safe_step(
                        "tool_error",
                        tool=decision.tool,
                        fingerprint=fingerprint,
                        error_code=result.error.code if result.error else "tool_error",
                    ),
                ],
            }

        records = _deduplicate_records(
            [*state.get("research_records", []), *result.records]
        )
        evidence = _deduplicate_evidence(
            [*state.get("tool_evidence", []), *result.evidence]
        )
        logger.info(
            "researcher_tool_result",
            extra={
                **self._log_context(
                    state,
                    int(state.get("research_step_count", 0)) + 1,
                    "tool_result",
                ),
                "tool": decision.tool.value,
                "record_ids": sorted(record_ids),
                "returned": result.meta.returned,
                "truncated": result.meta.truncated,
            },
        )
        return {
            "research_records": records,
            "tool_evidence": evidence,
            "research_transcript": transcript,
            "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            "executed_tool_fingerprints": fingerprints,
            "research_complete": False,
            "run_status": "running",
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step(
                    "tool_result",
                    tool=decision.tool,
                    relation_types=(
                        [item.value for item in request.relation_types]
                        if isinstance(request, RelationsRequest)
                        else []
                    ),
                    record_ids=record_ids,
                    fingerprint=fingerprint,
                    count=result.meta.returned,
                    resolution_strategy=(
                        request.match_mode.value
                        if isinstance(request, (PersonsRequest, CompaniesRequest))
                        else None
                    ),
                    resolution_version=(
                        ENTITY_MATCH_ALGORITHM_VERSION
                        if isinstance(request, (PersonsRequest, CompaniesRequest))
                        else None
                    ),
                ),
            ],
        }

    def _matching_contracts(
        self,
        state: AgentState,
        tool: ToolName,
        raw_arguments: dict[str, Any],
        request: BaseModel,
    ) -> list[_TaskContract]:
        matches: list[_TaskContract] = []
        for contract in self._snapshot(state).contracts:
            if contract.tool is not tool:
                continue
            if tool is ToolName.RELATIONS:
                assert isinstance(request, RelationsRequest)
                if contract.arguments is not None and self._relation_request_matches_contract(
                    raw_arguments, request, contract.arguments
                ):
                    matches.append(contract)
                continue
            if self._entity_request_matches_contract(request, contract):
                matches.append(contract)
        return matches

    def _entity_request_matches_contract(
        self, request: BaseModel, contract: _TaskContract
    ) -> bool:
        if isinstance(request, PersonsRequest):
            ids = [str(item) for item in request.person_ids]
            query = request.query
            attributes = [item.value for item in request.attributes]
        elif isinstance(request, CompaniesRequest):
            ids = [str(item) for item in request.company_ids]
            query = request.query
            attributes = [item.value for item in request.attributes]
        else:
            return False
        if set(contract.requested_attributes) - set(attributes):
            return False
        if contract.allowed_entity_ids:
            return (
                not query
                and set(ids) == set(contract.allowed_entity_ids)
                and request.match_mode is MatchMode.EXACT
            )
        return (
            not ids
            and (not contract.candidate_queries or query in contract.candidate_queries)
            and request.match_mode in {MatchMode.EXACT, MatchMode.FUZZY}
        )

    @staticmethod
    def _relation_request_matches_contract(
        raw_arguments: dict[str, Any],
        request: RelationsRequest,
        contract: dict[str, Any],
    ) -> bool:
        if set(raw_arguments) != set(_RELATION_FIELDS):
            return False
        normalized = request.model_dump(mode="json")
        unordered_fields = {
            "subject_ids",
            "object_ids",
            "relation_types",
            "raw_relation_types",
        }
        return all(
            (
                set(normalized.get(field) or []) == set(contract.get(field) or [])
                if field in unordered_fields
                else normalized.get(field) == contract.get(field)
            )
            for field in _RELATION_FIELDS
        )

    def _validate_trusted_ids(
        self, state: AgentState, tool: ToolName, request: BaseModel
    ) -> None:
        trusted = set(self._snapshot(state).bindings.values())
        if tool is ToolName.PERSONS:
            ids = set(str(item) for item in request.person_ids)  # type: ignore[attr-defined]
        elif tool is ToolName.COMPANIES:
            ids = set(str(item) for item in request.company_ids)  # type: ignore[attr-defined]
        else:
            ids = {
                *[str(item) for item in request.subject_ids],  # type: ignore[attr-defined]
                *[str(item) for item in request.object_ids],  # type: ignore[attr-defined]
            }
        if ids - trusted:
            raise ValueError("tool arguments contain unverified entity IDs")

    def _snapshot(self, state: AgentState) -> _TaskSnapshot:
        planner = PlannerDecision.model_validate(state.get("planner_decision"))
        records = list(state.get("research_records", []))
        records_by_id = {
            str(record.get("id")): record
            for record in records
            if record.get("id") is not None
        }
        bindings = {
            index: str(reference.context_entity_id)
            for index, reference in enumerate(planner.entity_references)
            if reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
            and reference.context_entity_id is not None
        }
        completed: set[str] = set()
        task_receipts: dict[str, list[dict[str, Any]]] = {
            task.task_id: [] for task in planner.research_tasks
        }
        tasks_by_id = {task.task_id: task for task in planner.research_tasks}

        for receipt in state.get("research_transcript", []):
            if not receipt.get("executed", True) or not receipt.get("success"):
                continue
            meta = receipt.get("meta") or {}
            if bool(meta.get("truncated")):
                continue
            try:
                tool = ToolName(str(receipt.get("tool")))
            except ValueError:
                continue
            explicit_task_ids = [
                str(item)
                for item in receipt.get("task_ids", [])
                if str(item) in tasks_by_id
            ]
            if tool in {ToolName.PERSONS, ToolName.COMPANIES}:
                matched_refs = self._bind_entity_receipt(
                    planner,
                    receipt,
                    records_by_id,
                    bindings,
                    tool,
                )
                candidate_tasks = explicit_task_ids or [
                    task.task_id
                    for task in planner.research_tasks
                    if task.tool is tool
                    and set(task.depends_on) <= completed
                    and set(self._task_reference_indexes(task)) & set(matched_refs)
                ]
                for task_id in candidate_tasks:
                    task_receipts[task_id].append(receipt)
                for task_id in candidate_tasks:
                    task = tasks_by_id[task_id]
                    indexes = self._task_reference_indexes(task)
                    if (not indexes or set(indexes) <= bindings.keys()) and self._entity_receipt_satisfies_task(
                        task, receipt, bindings, records_by_id
                    ):
                        completed.add(task_id)
                continue

            ready_relation_tasks = [
                task
                for task in planner.research_tasks
                if task.tool is ToolName.RELATIONS
                and set(task.depends_on) <= completed
                and set(self._task_reference_indexes(task)) <= bindings.keys()
            ]
            candidate_tasks = explicit_task_ids or [
                task.task_id
                for task in ready_relation_tasks
                if self._receipt_matches_relation_task(task, receipt, bindings)
            ]
            for task_id in candidate_tasks:
                task = tasks_by_id[task_id]
                if task not in ready_relation_tasks:
                    continue
                if self._receipt_matches_relation_task(task, receipt, bindings):
                    task_receipts[task_id].append(receipt)
                    completed.add(task_id)

        contracts = self._ready_contracts(planner, completed, bindings, state)
        depended_on = {
            dependency
            for task in planner.research_tasks
            for dependency in task.depends_on
        }
        terminal_task_ids = {
            task.task_id
            for task in planner.research_tasks
            if task.task_id not in depended_on
        }
        selected_records, result_nonempty = self._project_task_results(
            planner,
            terminal_task_ids,
            task_receipts,
            records_by_id,
            bindings,
        )
        return _TaskSnapshot(
            planner=planner,
            bindings=bindings,
            task_receipts=task_receipts,
            completed=completed,
            contracts=contracts,
            terminal_task_ids=terminal_task_ids,
            selected_records=selected_records,
            result_nonempty=result_nonempty,
        )

    @staticmethod
    def _task_reference_indexes(task: ResearchTask) -> list[int]:
        return list(
            dict.fromkeys(
                [*task.subject_reference_indexes, *task.object_reference_indexes]
            )
        )

    def _bind_entity_receipt(
        self,
        planner: PlannerDecision,
        receipt: dict[str, Any],
        records_by_id: dict[str, dict[str, Any]],
        bindings: dict[int, str],
        tool: ToolName,
    ) -> list[int]:
        meta = receipt.get("meta") or {}
        if meta.get("ambiguous") or meta.get("requires_clarification"):
            return []
        expected_type = (
            NodeType.PERSON.value
            if tool is ToolName.PERSONS
            else NodeType.COMPANY.value
        )
        entity_records = [
            records_by_id[record_id]
            for record_id in receipt.get("record_ids", [])
            if record_id in records_by_id
            and records_by_id[record_id].get("record_kind") == "entity"
            and records_by_id[record_id].get("entity_type") == expected_type
        ]
        if len(entity_records) != 1:
            return []
        entity_id = str(entity_records[0]["id"])
        arguments = receipt.get("arguments") or {}
        query = arguments.get("query")
        supplied_ids = {
            str(item)
            for key in ("person_ids", "company_ids")
            for item in arguments.get(key, [])
        }
        explicit_indexes = [
            int(item)
            for item in receipt.get("reference_indexes", [])
            if isinstance(item, int) and 0 <= item < len(planner.entity_references)
        ]
        candidates = explicit_indexes or [
            index
            for index, reference in enumerate(planner.entity_references)
            if NodeType(expected_type) in reference.expected_types
            and (
                query in {reference.canonical_name, reference.mention}
                or entity_id in supplied_ids
            )
        ]
        matched: list[int] = []
        for index in candidates:
            previous = bindings.get(index)
            if previous is not None and previous != entity_id:
                continue
            bindings[index] = entity_id
            matched.append(index)
        return matched

    def _entity_receipt_satisfies_task(
        self,
        task: ResearchTask,
        receipt: dict[str, Any],
        bindings: dict[int, str],
        records_by_id: dict[str, dict[str, Any]],
    ) -> bool:
        ids = {
            str(item) for item in receipt.get("record_ids", []) if str(item) in records_by_id
        }
        indexes = self._task_reference_indexes(task)
        if indexes and any(bindings.get(index) not in ids for index in indexes):
            # A name-resolution task may span several one-entity receipts; the task
            # is complete when every reference has a successful receipt assigned.
            return all(bindings.get(index) is not None for index in indexes)
        return bool(ids) or not indexes

    def _receipt_matches_relation_task(
        self,
        task: ResearchTask,
        receipt: dict[str, Any],
        bindings: dict[int, str],
    ) -> bool:
        expected = self._relation_arguments(task, bindings)
        try:
            request = RelationsRequest.model_validate(receipt.get("arguments") or {})
        except ValidationError:
            return False
        return self._relation_request_matches_contract(
            receipt.get("arguments") or {}, request, expected
        )

    def _ready_contracts(
        self,
        planner: PlannerDecision,
        completed: set[str],
        bindings: dict[int, str],
        state: AgentState,
    ) -> list[_TaskContract]:
        raw: list[_TaskContract] = []
        for task in planner.research_tasks:
            if task.task_id in completed or not set(task.depends_on) <= completed:
                continue
            indexes = self._task_reference_indexes(task)
            if task.tool is ToolName.RELATIONS:
                if not set(indexes) <= bindings.keys():
                    continue
                raw.append(
                    _TaskContract(
                        task_ids=(task.task_id,),
                        tool=task.tool,
                        reference_indexes=tuple(indexes),
                        arguments=self._relation_arguments(task, bindings),
                    )
                )
                continue
            expected_type = (
                NodeType.PERSON
                if task.tool is ToolName.PERSONS
                else NodeType.COMPANY
            )
            typed_indexes = [
                index
                for index in indexes
                if expected_type in planner.entity_references[index].expected_types
            ]
            unresolved = [index for index in typed_indexes if index not in bindings]
            if unresolved:
                for index in unresolved:
                    reference = planner.entity_references[index]
                    preferred = reference.canonical_name or reference.mention
                    raw.append(
                        _TaskContract(
                            task_ids=(task.task_id,),
                            tool=task.tool,
                            reference_indexes=(index,),
                            candidate_queries=(preferred,),
                            requested_attributes=tuple(task.requested_attributes),
                        )
                    )
            else:
                raw.append(
                    _TaskContract(
                        task_ids=(task.task_id,),
                        tool=task.tool,
                        reference_indexes=tuple(typed_indexes),
                        allowed_entity_ids=tuple(
                            sorted(bindings[index] for index in typed_indexes)
                        ),
                        requested_attributes=tuple(task.requested_attributes),
                    )
                )

        grouped: dict[str, _TaskContract] = {}
        for contract in raw:
            identity = json.dumps(
                {
                    "tool": contract.tool.value,
                    "references": contract.reference_indexes,
                    "arguments": contract.arguments,
                    "queries": contract.candidate_queries,
                    "ids": contract.allowed_entity_ids,
                    "attrs": contract.requested_attributes,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            previous = grouped.get(identity)
            if previous is None:
                grouped[identity] = contract
            else:
                grouped[identity] = _TaskContract(
                    task_ids=tuple(dict.fromkeys([*previous.task_ids, *contract.task_ids])),
                    tool=contract.tool,
                    reference_indexes=contract.reference_indexes,
                    arguments=contract.arguments,
                    candidate_queries=contract.candidate_queries,
                    allowed_entity_ids=contract.allowed_entity_ids,
                    requested_attributes=contract.requested_attributes,
                )
        return list(grouped.values())

    @staticmethod
    def _relation_arguments(
        task: ResearchTask, bindings: dict[int, str]
    ) -> dict[str, Any]:
        direction = {
            ResearchDirection.OUTGOING: RelationDirection.OUTGOING.value,
            ResearchDirection.INCOMING: RelationDirection.INCOMING.value,
            ResearchDirection.ANY: RelationDirection.ANY.value,
        }[task.direction]
        return {
            "subject_ids": sorted(
                {bindings[index] for index in task.subject_reference_indexes}
            ),
            "object_ids": sorted(
                {bindings[index] for index in task.object_reference_indexes}
            ),
            "relation_types": sorted(
                {item.value for item in task.relation_types}
            ),
            "raw_relation_types": sorted(set(task.raw_relation_types)),
            "direction": direction,
            "include_endpoints": True,
            "limit": 200,
        }

    def _project_task_results(
        self,
        planner: PlannerDecision,
        terminal_task_ids: set[str],
        task_receipts: dict[str, list[dict[str, Any]]],
        records_by_id: dict[str, dict[str, Any]],
        bindings: dict[int, str],
    ) -> tuple[list[dict[str, Any]], bool]:
        tasks = {task.task_id: task for task in planner.research_tasks}
        terminal_relation_ids = {
            task_id
            for task_id in terminal_task_ids
            if tasks[task_id].tool is ToolName.RELATIONS
        }
        relation_records_by_task: dict[str, list[dict[str, Any]]] = {}
        terminal_entity_record_ids: set[str] = set()
        for task_id in terminal_task_ids:
            task = tasks[task_id]
            receipt_records = {
                str(record_id)
                for receipt in task_receipts.get(task_id, [])
                for record_id in receipt.get("record_ids", [])
            }
            if task.tool is ToolName.RELATIONS:
                relation_records_by_task[task_id] = [
                    records_by_id[record_id]
                    for record_id in receipt_records
                    if record_id in records_by_id
                    and records_by_id[record_id].get("record_kind") == "relation"
                ]
            else:
                terminal_entity_record_ids.update(
                    record_id
                    for record_id in receipt_records
                    if record_id in records_by_id
                    and records_by_id[record_id].get("record_kind") == "entity"
                )

        relations = self._merge_relation_results(
            planner,
            terminal_relation_ids,
            relation_records_by_task,
            tasks,
            bindings,
            records_by_id,
        )
        relation_ids = {str(record["id"]) for record in relations}
        endpoint_ids = {
            str(endpoint)
            for record in relations
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        bound_ids = {
            bindings[index]
            for task_id in terminal_task_ids
            for index in self._task_reference_indexes(tasks[task_id])
            if index in bindings
        }
        selected_ids = {
            *relation_ids,
            *endpoint_ids,
            *terminal_entity_record_ids,
            *bound_ids,
        }
        selected = [
            record for record_id, record in records_by_id.items() if record_id in selected_ids
        ]
        result_nonempty = bool(relations) if terminal_relation_ids else bool(
            terminal_entity_record_ids
        )
        return _deduplicate_records(selected), result_nonempty

    def _merge_relation_results(
        self,
        planner: PlannerDecision,
        terminal_relation_ids: set[str],
        records_by_task: dict[str, list[dict[str, Any]]],
        tasks: dict[str, ResearchTask],
        bindings: dict[int, str],
        records_by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        filtered_by_task: dict[str, list[dict[str, Any]]] = {}
        for task_id in terminal_relation_ids:
            task = tasks[task_id]
            task_records: list[dict[str, Any]] = []
            for record in records_by_task.get(task_id, []):
                if self._relation_matches_task_target(
                    task, record, bindings, records_by_id
                ):
                    task_records.append(record)
            filtered_by_task[task_id] = task_records

        all_relations = _deduplicate_records(
            [
                record
                for task_id in terminal_relation_ids
                for record in filtered_by_task.get(task_id, [])
            ]
        )
        if planner.result_merge is not ResultMergeStrategy.INTERSECTION or len(
            terminal_relation_ids
        ) < 2:
            return all_relations

        neighbors_by_task: list[set[str]] = []
        all_subjects = {
            bindings[index]
            for task_id in terminal_relation_ids
            for index in tasks[task_id].subject_reference_indexes
            if index in bindings
        }
        for task_id in sorted(terminal_relation_ids):
            task_subjects = {
                bindings[index]
                for index in tasks[task_id].subject_reference_indexes
                if index in bindings
            }
            neighbors = {
                str(endpoint)
                for record in filtered_by_task.get(task_id, [])
                for endpoint in (record.get("source"), record.get("target"))
                if endpoint is not None and str(endpoint) not in task_subjects
            }
            neighbors_by_task.append(neighbors)
        common = set.intersection(*neighbors_by_task) if neighbors_by_task else set()
        return [
            record
            for record in all_relations
            if (
                str(record.get("source")) in common
                or str(record.get("target")) in common
            )
            and (
                str(record.get("source")) in all_subjects
                or str(record.get("target")) in all_subjects
            )
        ]

    @staticmethod
    def _relation_matches_task_target(
        task: ResearchTask,
        record: dict[str, Any],
        bindings: dict[int, str],
        records_by_id: dict[str, dict[str, Any]],
    ) -> bool:
        allowed_types = {item.value for item in task.target_types}
        if not allowed_types:
            return True
        subject_ids = {
            bindings[index]
            for index in task.subject_reference_indexes
            if index in bindings
        }
        endpoints = [
            str(record.get("source", "")),
            str(record.get("target", "")),
        ]
        candidates = [endpoint for endpoint in endpoints if endpoint not in subject_ids]
        if not candidates:
            candidates = endpoints
        return any(
            str(records_by_id.get(endpoint, {}).get("entity_type", ""))
            in allowed_types
            for endpoint in candidates
        )

    def _finish(
        self, state: AgentState, decision: ResearcherDecision
    ) -> dict[str, Any]:
        snapshot = self._snapshot(state)
        if not snapshot.all_complete:
            raise ValueError("not all Planner tasks have complete successful receipts")
        expected_action = (
            ResearchAction.FINISH
            if snapshot.result_nonempty
            else ResearchAction.NO_RESULTS
        )
        if decision.action is not expected_action:
            raise ValueError("lifecycle action disagrees with merged task results")

        selected_ids = sorted(
            str(record["id"])
            for record in snapshot.selected_records
            if record.get("id") is not None
        )
        if decision.selected_record_ids and set(decision.selected_record_ids) != set(
            selected_ids
        ):
            raise ValueError("scripted selected IDs disagree with receipt projection")
        evidence_by_id = {item.id: item for item in state.get("tool_evidence", [])}
        referenced_evidence = {
            str(evidence_id)
            for record in snapshot.selected_records
            for evidence_id in record.get("evidence_ids", [])
        }
        if referenced_evidence - evidence_by_id.keys():
            raise ValueError("selected records reference missing Evidence")

        signature = self._derive_signature(state, snapshot)
        if snapshot.result_nonempty:
            validate_signature_records(
                signature,
                snapshot.selected_records,
                state.get("research_records", []),
            )
        query_resolved = {
            reference.mention: snapshot.bindings[index]
            for index, reference in enumerate(snapshot.planner.entity_references)
            if reference.source is EntityReferenceSource.CURRENT_QUERY
            and index in snapshot.bindings
        }
        if snapshot.result_nonempty:
            focus = expected_focus_entity_ids(
                signature,
                snapshot.selected_records,
                state.get("research_records", []),
            )
        else:
            focus = sorted(
                {
                    snapshot.bindings[index]
                    for task in snapshot.planner.research_tasks
                    for index in task.subject_reference_indexes
                    if index in snapshot.bindings
                }
            )
        logger.info(
            "researcher_finished",
            extra={
                **self._log_context(
                    state,
                    int(state.get("research_step_count", 0)) + 1,
                    expected_action.value,
                ),
                "selected_record_ids": selected_ids,
                "no_match": not snapshot.result_nonempty,
            },
        )
        return {
            "selected_record_ids": selected_ids,
            "query_signature": signature,
            "query_resolved_entities": query_resolved,
            "turn_focus_entity_ids": focus,
            "research_complete": True,
            "no_match": not snapshot.result_nonempty,
            "needs_replan": False,
            "current_replan_reason": None,
            "run_status": "success",
        }

    def _derive_signature(
        self, state: AgentState, snapshot: _TaskSnapshot
    ) -> QuerySignature:
        planner = snapshot.planner
        terminal_tasks = [
            task
            for task in planner.research_tasks
            if task.task_id in snapshot.terminal_task_ids
        ]
        relation_tasks = [
            task for task in terminal_tasks if task.tool is ToolName.RELATIONS
        ]
        # Result projection intentionally remains terminal-task-only.  Signature
        # auditing, however, must retain every completed relation scope in the DAG,
        # including prerequisite probes such as an explicit ``controls`` query that
        # returned zero rows before a strong-association fallback ran.
        completed_relation_tasks = [
            task
            for task in planner.research_tasks
            if task.tool is ToolName.RELATIONS and task.task_id in snapshot.completed
        ]
        relevant_tasks = relation_tasks or terminal_tasks
        subject_ids = {
            snapshot.bindings[index]
            for task in relevant_tasks
            for index in task.subject_reference_indexes
            if index in snapshot.bindings
        }
        explicit_objects = {
            snapshot.bindings[index]
            for task in relevant_tasks
            for index in task.object_reference_indexes
            if index in snapshot.bindings
        }
        selected_relations = [
            record
            for record in snapshot.selected_records
            if record.get("record_kind") == "relation"
        ]
        selected_relation_ids = {str(record.get("id")) for record in selected_relations}
        endpoints = {
            str(endpoint)
            for record in selected_relations
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        if explicit_objects:
            object_ids = set(explicit_objects)
        elif planner.result_merge is ResultMergeStrategy.UNION:
            # A seed may itself be a company neighbour in another independent
            # union task (for example company A is related to person B).  Remove
            # only each task's own subjects, not every seed globally.
            object_ids: set[str] = set()
            records_by_id = {
                str(record.get("id")): record
                for record in state.get("research_records", [])
                if record.get("id") is not None
            }
            for task in relation_tasks:
                task_subjects = {
                    snapshot.bindings[index]
                    for index in task.subject_reference_indexes
                    if index in snapshot.bindings
                }
                task_relation_ids = {
                    str(record_id)
                    for receipt in snapshot.task_receipts.get(task.task_id, [])
                    for record_id in receipt.get("record_ids", [])
                    if str(record_id) in selected_relation_ids
                    and self._relation_matches_task_target(
                        task,
                        records_by_id.get(str(record_id), {}),
                        snapshot.bindings,
                        records_by_id,
                    )
                }
                object_ids.update(
                    str(endpoint)
                    for record_id in task_relation_ids
                    for endpoint in (
                        records_by_id.get(record_id, {}).get("source"),
                        records_by_id.get(record_id, {}).get("target"),
                    )
                    if endpoint is not None and str(endpoint) not in task_subjects
                )
        else:
            object_ids = endpoints - subject_ids
        records_by_id = {
            str(record.get("id")): record
            for record in state.get("research_records", [])
            if record.get("id") is not None
        }

        def task_has_relations(task: ResearchTask) -> bool:
            return any(
                records_by_id.get(str(record_id), {}).get("record_kind") == "relation"
                for receipt in snapshot.task_receipts.get(task.task_id, [])
                for record_id in receipt.get("record_ids", [])
            )

        requested_relation_types = {
            relation_type
            for task in completed_relation_tasks
            for relation_type in task.relation_types
        }
        nonempty_relation_tasks = [
            task for task in completed_relation_tasks if task_has_relations(task)
        ]
        effective_relation_types = {
            relation_type
            for task in nonempty_relation_tasks
            for relation_type in task.relation_types
        }
        # A non-empty blank-filter task is an effective complete direct scope.  It
        # therefore dominates narrower typed filters in the evidence signature.
        if any(
            not task.relation_types and not task.raw_relation_types
            for task in nonempty_relation_tasks
        ):
            effective_relation_types = set()
        raw_qualifiers = {
            item
            for task in completed_relation_tasks
            for item in task.raw_relation_types
        }
        empty_types = {
            relation_type
            for task in completed_relation_tasks
            if not task_has_relations(task)
            for relation_type in task.relation_types
        }
        if planner.intent is Intent.FIND_CONTROLLED_COMPANIES:
            control_policy = (
                ControlQueryPolicy.EXPLICIT_ONLY
                if completed_relation_tasks
                and all(
                    set(task.relation_types) == {RelationType.CONTROLS}
                    and not task.raw_relation_types
                    for task in completed_relation_tasks
                )
                else ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS
            )
        else:
            control_policy = ControlQueryPolicy.NOT_APPLICABLE
        context_ids = [
            str(reference.context_entity_id)
            for reference in planner.entity_references
            if reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
            and reference.context_entity_id is not None
        ]
        return QuerySignature(
            version=self.query_signature_version,
            intent=planner.intent,
            subject_ids=sorted(subject_ids),
            object_ids=sorted(object_ids),
            relation_types=sorted(
                effective_relation_types, key=lambda item: item.value
            ),
            requested_relation_types=sorted(
                requested_relation_types, key=lambda item: item.value
            ),
            effective_relation_types=sorted(
                effective_relation_types, key=lambda item: item.value
            ),
            raw_relation_qualifiers=sorted(raw_qualifiers),
            verified_empty_relation_types=sorted(
                empty_types, key=lambda item: item.value
            ),
            target_types=list(
                dict.fromkeys(
                    node_type for task in relevant_tasks for node_type in task.target_types
                )
            ),
            requested_attributes=list(
                dict.fromkeys(
                    attribute
                    for task in relevant_tasks
                    for attribute in task.requested_attributes
                )
            ),
            context_entity_ids=context_ids,
            result_merge=planner.result_merge,
            control_policy=control_policy,
            entity_match_version=ENTITY_MATCH_ALGORITHM_VERSION,
            locale=str(state.get("locale", "zh-CN")),
        )

    def _payload(self, state: AgentState) -> dict[str, Any]:
        snapshot = self._snapshot(state)
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "plan": {
                "intent": snapshot.planner.intent.value,
                "result_merge": snapshot.planner.result_merge.value,
                "entity_references": [
                    {
                        "index": index,
                        "mention": reference.mention,
                        "canonical_name": reference.canonical_name,
                        "source": reference.source.value,
                        "expected_types": [item.value for item in reference.expected_types],
                        "context_entity_id": reference.context_entity_id,
                    }
                    for index, reference in enumerate(snapshot.planner.entity_references)
                ],
                "research_tasks": [
                    task.model_dump(mode="json")
                    for task in snapshot.planner.research_tasks
                ],
            },
            "verified_bindings": {
                str(index): entity_id for index, entity_id in snapshot.bindings.items()
            },
            "task_status": [
                {
                    "task_id": task.task_id,
                    "status": (
                        "completed"
                        if task.task_id in snapshot.completed
                        else "blocked"
                        if not set(task.depends_on) <= snapshot.completed
                        else "ready"
                    ),
                }
                for task in snapshot.planner.research_tasks
            ],
            "ready_task_contracts": [
                contract.public_summary() for contract in snapshot.contracts
            ],
            "verified_receipts": self._receipt_summaries(state),
            "contract_feedback": state.get("current_replan_reason"),
            "counters": {
                "model_actions": state.get("research_step_count", 0),
                "tool_calls": state.get("tool_call_count", 0),
                "max_tool_calls": self.max_tool_calls,
            },
        }

    @staticmethod
    def _receipt_summaries(state: AgentState) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for receipt in state.get("research_transcript", []):
            meta = receipt.get("meta") or {}
            summaries.append(
                {
                    "task_ids": list(receipt.get("task_ids", [])),
                    "tool": receipt.get("tool"),
                    "success": bool(receipt.get("success")),
                    "executed": bool(receipt.get("executed", True)),
                    "record_ids": list(receipt.get("record_ids", [])),
                    "returned": meta.get("returned", 0),
                    "truncated": bool(meta.get("truncated", False)),
                    "error_code": receipt.get("error_code"),
                }
            )
        return summaries[-20:]

    def _tool_contract_failure(
        self,
        state: AgentState,
        decision: ResearcherDecision,
        error_code: str,
    ) -> dict[str, Any]:
        retry = int(state.get("researcher_contract_retry_count", 0))
        first = retry == 0
        transcript = list(state.get("research_transcript", []))
        if decision.tool is not None:
            transcript.append(
                self._receipt(
                    state,
                    decision.tool,
                    decision.arguments,
                    [],
                    executed=False,
                    success=False,
                    error_code=error_code,
                )
            )
        return {
            "research_transcript": transcript,
            "researcher_contract_retry_count": retry + 1,
            "run_status": "running" if first else "failed",
            "research_complete": False,
            "needs_replan": False,
            "llm_errors": (
                list(state.get("llm_errors", []))
                if first
                else [
                    *state.get("llm_errors", []),
                    "Researcher returned a response that violated its typed contract",
                ]
            ),
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step("contract_rejected", error_code=error_code),
            ],
        }

    def _contract_failure(
        self,
        state: AgentState,
        counters: dict[str, Any],
        route: list[str],
        *,
        error_code: str,
    ) -> dict[str, Any]:
        retry = int(state.get("researcher_contract_retry_count", 0))
        first = retry == 0
        return {
            **counters,
            "researcher_contract_retry_count": retry + 1,
            "run_status": "running" if first else "failed",
            "research_complete": False,
            "needs_replan": False,
            "route_history": route,
            "llm_errors": (
                list(state.get("llm_errors", []))
                if first
                else [
                    *state.get("llm_errors", []),
                    "Researcher returned a response that violated its typed contract",
                ]
            ),
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step("contract_rejected", error_code=error_code),
            ],
        }

    def _completion_failure(
        self, state: AgentState, base: dict[str, Any]
    ) -> dict[str, Any]:
        retry = int(state.get("researcher_contract_retry_count", 0))
        first = retry == 0
        return {
            **base,
            "researcher_contract_retry_count": retry + 1,
            "run_status": "running" if first else "failed",
            "research_complete": False,
            "needs_replan": False,
            "llm_errors": (
                list(state.get("llm_errors", []))
                if first
                else [
                    *state.get("llm_errors", []),
                    "Researcher completion violated the verified receipt contract",
                ]
            ),
            "agent_steps": [
                *base["agent_steps"],
                self._safe_step(
                    "completion_rejected", error_code="invalid_verified_completion"
                ),
            ],
        }

    def _synthetic_tool_failure(
        self,
        state: AgentState,
        tool: ToolName,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> TypedToolResult:
        return TypedToolResult(
            success=False,
            tool=tool,
            provider="local-mock",
            data_version=str(state.get("data_version", "unknown")),
            records=[],
            evidence=[],
            elapsed_ms=0,
            error=ToolError(
                tool=tool,
                code=code,
                message=message,
                retryable=retryable,
            ),
            meta=ToolResultMeta(),
        )

    @staticmethod
    def _receipt(
        state: AgentState,
        tool: ToolName,
        arguments: dict[str, Any],
        contracts: list[_TaskContract],
        *,
        executed: bool,
        success: bool,
        error_code: str | None,
        fingerprint: str | None = None,
        record_ids: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        data_version: str | None = None,
    ) -> dict[str, Any]:
        return {
            "step": int(state.get("research_step_count", 0)) + 1,
            "task_ids": list(
                dict.fromkeys(
                    task_id for contract in contracts for task_id in contract.task_ids
                )
            ),
            "reference_indexes": list(
                dict.fromkeys(
                    index
                    for contract in contracts
                    for index in contract.reference_indexes
                )
            ),
            "tool": tool.value,
            "arguments": arguments,
            "success": success,
            "executed": executed,
            "record_ids": record_ids or [],
            "meta": meta or {},
            "data_version": data_version or state.get("data_version"),
            "argument_fingerprint": fingerprint,
            "error_code": error_code,
        }

    @staticmethod
    def _tool_call_fingerprint(tool: ToolName, arguments: dict[str, Any]) -> str:
        payload = json.dumps(
            {"tool": tool.value, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_step(
        action: str,
        *,
        tool: ToolName | None = None,
        relation_types: list[str] | None = None,
        record_ids: list[str] | None = None,
        fingerprint: str | None = None,
        count: int = 0,
        error_code: str | None = None,
        resolution_strategy: str | None = None,
        resolution_version: str | None = None,
    ) -> dict[str, Any]:
        return {
            "role": "researcher",
            "action": action,
            "tool": tool.value if tool else None,
            "relation_types": sorted(relation_types or []),
            "result_merge": None,
            "resolution_strategy": resolution_strategy,
            "resolution_version": resolution_version,
            "record_ids": sorted(record_ids or []),
            "argument_fingerprint": fingerprint,
            "count": count,
            "error_code": error_code,
        }

    @staticmethod
    def _log_context(
        state: AgentState, research_step: int, action: str
    ) -> dict[str, Any]:
        return {
            "event": "researcher_action",
            "request_id": state.get("request_id"),
            "conversation_id": state.get("conversation_id"),
            "research_step": research_step,
            "action": action,
        }

    @staticmethod
    def _terminal_failure(
        state: AgentState, reason: str, route: list[str]
    ) -> dict[str, Any]:
        return {
            "run_status": "failed",
            "research_complete": False,
            "research_failure_reason": reason,
            "route_history": route,
            "agent_steps": [
                *state.get("agent_steps", []),
                Researcher._safe_step("fail", error_code="research_limit_exhausted"),
            ],
        }


@dataclass(slots=True)
class ResultGate:
    """Convert Researcher lifecycle state into one bounded StateGraph route."""

    max_steps: int = 12
    hard_max_steps: int | None = None
    max_replans: int = 2
    retry_step_allowance: int = 3

    def _hard_step_limit(self) -> int:
        return max(self.max_steps, self.hard_max_steps or self.max_steps)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "result_gate"]
        if state.get("needs_replan") and int(state.get("replan_count", 0)) >= self.max_replans:
            return {
                "needs_replan": False,
                "run_status": "failed",
                "research_failure_reason": "Researcher exhausted the configured replan limit",
                "route_history": route,
            }
        if state.get("needs_replan"):
            if int(state.get("research_step_count", 0)) >= self._hard_step_limit():
                return {
                    "needs_replan": False,
                    "run_status": "failed",
                    "research_failure_reason": "Researcher exhausted the absolute model-iteration limit",
                    "route_history": route,
                }
            return {"route_history": route}
        effective = _effective_research_step_limit(
            state,
            base_steps=self.max_steps,
            hard_max_steps=self._hard_step_limit(),
            max_replans=self.max_replans,
            retry_step_allowance=self.retry_step_allowance,
        )
        if state.get("run_status") == "running" and int(
            state.get("research_step_count", 0)
        ) >= effective:
            return {
                "run_status": "failed",
                "research_failure_reason": "Researcher exhausted the current model-step budget",
                "route_history": route,
            }
        return {"route_history": route}


def route_after_result_gate(state: AgentState) -> str:
    if state.get("needs_replan") and state.get("run_status") != "failed":
        return "replan"
    if state.get("run_status") == "success" and state.get("research_complete"):
        return "no_match" if state.get("no_match") else "valid"
    if state.get("run_status") == "running":
        return "research"
    return "error"
