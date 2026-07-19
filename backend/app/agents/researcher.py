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
from app.ids import normalize_query
from app.llm import ModelClient, ModelInvocationError, NativeToolCall
from app.schemas import (
    ControlQueryPolicy,
    EntityReferenceSource,
    Evidence,
    GoalResultStatus,
    GraphPayload,
    Intent,
    NodeType,
    PlannerDecision,
    QueryGoalSignature,
    QuerySignature,
    RelationType,
    ResearchAction,
    ResearchDirection,
    ResearchGoal,
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
    scope_entity_openai_parameters,
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
    query_rewrites: tuple[tuple[str, str], ...] = ()
    reference_queries: tuple[tuple[int, str], ...] = ()
    allowed_entity_ids: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()
    required_match_mode: MatchMode | None = None

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
            value["query_rewrites"] = [
                {
                    "original_query": original,
                    "rewritten_query": rewritten,
                }
                for original, rewritten in self.query_rewrites
            ]
            value["allowed_entity_ids"] = list(self.allowed_entity_ids)
            value["requested_attributes"] = list(self.requested_attributes)
            value["required_match_mode"] = (
                self.required_match_mode.value
                if self.required_match_mode is not None
                else None
            )
        return value


@dataclass(slots=True)
class _TaskSnapshot:
    planner: PlannerDecision
    bindings: dict[int, str]
    task_receipts: dict[str, list[dict[str, Any]]]
    goal_results: dict[str, "_GoalResult"]
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
class _GoalResult:
    goal: ResearchGoal
    complete: bool
    skipped_empty_input: bool
    subject_ids: set[str]
    explicit_object_ids: set[str]
    result_entity_ids: set[str]
    focus_entity_ids: set[str]
    selected_records: list[dict[str, Any]]
    nonempty: bool


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
    query_signature_version: int = 5

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

        step_counters = {
            "research_step_count": steps + 1,
            "researcher_invoked": True,
        }
        try:
            prepared = self._prepare_model_invocation(state)
        except (ValidationError, TypeError, ValueError):
            return self._contract_failure(
                state,
                step_counters,
                route,
                error_code="invalid_typed_contract",
            )

        # Count only a real provider attempt.  Building the request-local native
        # function Schema is a deterministic preflight and must not masquerade
        # as a model call when that local contract is invalid.
        counters = {
            **step_counters,
            "model_call_count": int(state.get("model_call_count", 0)) + 1,
            "researcher_model_calls": int(state.get("researcher_model_calls", 0)) + 1,
        }
        try:
            value = await self._invoke_prepared_model(prepared)
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

    def _prepare_model_invocation(
        self, state: AgentState
    ) -> tuple[dict[str, Any], Any, list[dict[str, Any]] | None]:
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
            return payload, native_call, definitions
        return payload, None, None

    async def _invoke_prepared_model(
        self,
        prepared: tuple[dict[str, Any], Any, list[dict[str, Any]] | None],
    ) -> ResearcherDecision | dict[str, Any]:
        payload, native_call, definitions = prepared
        if callable(native_call):
            if definitions is None:
                raise ValueError("native Researcher invocation requires definitions")
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
            elif tool in {ToolName.PERSONS, ToolName.COMPANIES}:
                self._scope_entity_schema(definition, contracts)
            result.append(definition)
        return result

    @staticmethod
    def _scope_entity_schema(
        definition: dict[str, Any], contracts: list[_TaskContract]
    ) -> None:
        if not contracts:
            raise ValueError("entity function schema requires a ready task contract")

        # A function name exposes one closed argument schema in an iteration.
        # Scope it to the first ready typed contract in Planner task order; any
        # independent contracts stay ready for the next graph iteration.  This
        # chooses no intent or entity—it only projects an existing task contract.
        contract = contracts[0]
        if contract.required_match_mode is None:
            raise ValueError("entity task contract requires a match mode")
        try:
            tool = ToolName(str(definition.get("name")))
        except ValueError as exc:
            raise ValueError("unknown entity function definition") from exc
        if tool not in {ToolName.PERSONS, ToolName.COMPANIES}:
            raise ValueError("entity schema scoping requires an entity tool")

        scope_entity_openai_parameters(
            definition.get("parameters", {}),
            id_field=(
                "person_ids" if tool is ToolName.PERSONS else "company_ids"
            ),
            match_mode=contract.required_match_mode,
            queries=contract.candidate_queries,
            query_rewrites=contract.query_rewrites,
            entity_ids=contract.allowed_entity_ids,
            required_attributes=contract.requested_attributes,
        )

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
            else {
                key: (
                    sorted(
                        value,
                        key=lambda item: json.dumps(
                            item,
                            sort_keys=True,
                            ensure_ascii=False,
                        ),
                    )
                    if key
                    in {
                        "queries",
                        "query_rewrites",
                        "person_ids",
                        "company_ids",
                        "attributes",
                    }
                    and isinstance(value, list)
                    else value
                )
                for key, value in dumped_arguments.items()
            }
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
            queries = list(request.lookup_queries)
            rewrites = [
                (item.original_query, item.rewritten_query)
                for item in request.query_rewrites
            ]
            attributes = [item.value for item in request.attributes]
        elif isinstance(request, CompaniesRequest):
            ids = [str(item) for item in request.company_ids]
            queries = list(request.lookup_queries)
            rewrites = [
                (item.original_query, item.rewritten_query)
                for item in request.query_rewrites
            ]
            attributes = [item.value for item in request.attributes]
        else:
            return False
        if set(contract.requested_attributes) - set(attributes):
            return False
        if contract.allowed_entity_ids:
            return (
                not queries
                and not rewrites
                and bool(ids)
                and set(ids) <= set(contract.allowed_entity_ids)
                and request.match_mode is MatchMode.EXACT
            )
        if contract.required_match_mode is None or (
            request.match_mode is not contract.required_match_mode
        ):
            return False
        if contract.required_match_mode is MatchMode.CROSS_LANGUAGE_EXACT:
            requested_rewrites = {
                (normalize_query(original), normalize_query(rewritten))
                for original, rewritten in rewrites
            }
            return (
                not ids
                and not request.query
                and not request.queries
                and bool(requested_rewrites)
                and requested_rewrites
                <= {
                    (normalize_query(original), normalize_query(rewritten))
                    for original, rewritten in contract.query_rewrites
                }
            )
        requested_queries = {normalize_query(item) for item in queries}
        return (
            not ids
            and not rewrites
            and bool(requested_queries)
            and requested_queries
            <= {normalize_query(item) for item in contract.candidate_queries}
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
        snapshot = self._snapshot(state)
        trusted = set(snapshot.bindings.values())
        # A dependent goal consumes entity IDs projected from a completed,
        # evidence-backed upstream goal.  Those IDs are just as trusted as
        # direct entity bindings; without this expansion a valid batched
        # location query is rejected even though every subject came from a
        # successful relation receipt.
        for goal_result in snapshot.goal_results.values():
            if goal_result.complete:
                trusted.update(goal_result.result_entity_ids)
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

        relation_receipts: list[dict[str, Any]] = []
        for receipt in state.get("research_transcript", []):
            if not receipt.get("executed", True) or not receipt.get("success"):
                continue
            meta = receipt.get("meta") or {}
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
                    if receipt not in task_receipts[task_id]:
                        task_receipts[task_id].append(receipt)
                for task_id in candidate_tasks:
                    task = tasks_by_id[task_id]
                    indexes = self._task_reference_indexes(task)
                    references_bound = not indexes or set(indexes) <= bindings.keys()
                    if references_bound and self._entity_receipts_satisfy_task(
                        task,
                        task_receipts[task_id],
                        bindings,
                        records_by_id,
                    ):
                        completed.add(task_id)
                continue
            if not bool(meta.get("truncated")):
                relation_receipts.append(receipt)

        # Relation tasks may consume the verified result set of an earlier goal.
        # Resolve them to a fixed point over successful receipts.  An empty dynamic
        # input is a completed, non-executable branch; dispatching subject_ids=[]
        # would mean "no filter" to the raw relation tool and could scan the full
        # fixture, so it is deliberately marked complete without a tool call.
        while True:
            progressed = False
            goal_results = self._build_goal_results(
                planner, completed, task_receipts, records_by_id, bindings
            )
            for task in planner.research_tasks:
                if (
                    task.task_id in completed
                    or task.tool is not ToolName.RELATIONS
                    or not set(task.depends_on) <= completed
                    or not set(self._task_reference_indexes(task)) <= bindings.keys()
                ):
                    continue
                if self._control_fallback_not_required(
                    task,
                    planner,
                    completed,
                    task_receipts,
                    records_by_id,
                ):
                    completed.add(task.task_id)
                    progressed = True
                    continue
                if self._task_has_verified_empty_input(task, goal_results, bindings):
                    completed.add(task.task_id)
                    progressed = True

            goal_results = self._build_goal_results(
                planner, completed, task_receipts, records_by_id, bindings
            )
            ready_relation_tasks = [
                task
                for task in planner.research_tasks
                if task.tool is ToolName.RELATIONS
                and task.task_id not in completed
                and set(task.depends_on) <= completed
                and set(self._task_reference_indexes(task)) <= bindings.keys()
                and self._task_dynamic_inputs_ready(task, goal_results)
            ]
            for receipt in relation_receipts:
                explicit_task_ids = [
                    str(item)
                    for item in receipt.get("task_ids", [])
                    if str(item) in tasks_by_id
                ]
                candidate_tasks = (
                    [tasks_by_id[item] for item in explicit_task_ids]
                    if explicit_task_ids
                    else ready_relation_tasks
                )
                for task in candidate_tasks:
                    if task not in ready_relation_tasks:
                        continue
                    if self._receipt_matches_relation_task(
                        task, receipt, bindings, goal_results
                    ):
                        if receipt not in task_receipts[task.task_id]:
                            task_receipts[task.task_id].append(receipt)
                        completed.add(task.task_id)
                        progressed = True
            if not progressed:
                break

        goal_results = self._build_goal_results(
            planner, completed, task_receipts, records_by_id, bindings
        )
        contracts = self._ready_contracts(
            planner, completed, bindings, state, goal_results
        )
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
        selected_records, result_nonempty = self._project_goal_results(
            planner, goal_results, records_by_id, bindings
        )
        return _TaskSnapshot(
            planner=planner,
            bindings=bindings,
            task_receipts=task_receipts,
            goal_results=goal_results,
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
        expected_type = (
            NodeType.PERSON.value
            if tool is ToolName.PERSONS
            else NodeType.COMPANY.value
        )
        entity_ids = {
            str(record_id)
            for record_id in receipt.get("record_ids", [])
            if record_id in records_by_id
            and records_by_id[record_id].get("record_kind") == "entity"
            and records_by_id[record_id].get("entity_type") == expected_type
        }
        arguments = receipt.get("arguments") or {}
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
        ]
        query_matches = {
            normalize_query(str(item.get("query", ""))): item
            for item in meta.get("query_matches", [])
            if isinstance(item, dict) and item.get("query")
        }
        matched: list[int] = []
        for index in candidates:
            reference = planner.entity_references[index]
            outcome = query_matches.get(normalize_query(reference.mention))
            if outcome is not None:
                outcome_ids = [str(item) for item in outcome.get("matched_entity_ids", [])]
                proven_ids = {
                    str(item.get("entity_id"))
                    for item in outcome.get("match_proofs", [])
                    if isinstance(item, dict) and item.get("entity_id")
                }
                if (
                    bool(outcome.get("ambiguous"))
                    or bool(outcome.get("truncated"))
                    or len(outcome_ids) != 1
                    or outcome_ids[0] not in entity_ids
                    or outcome_ids[0] not in proven_ids
                ):
                    continue
                entity_id = outcome_ids[0]
            else:
                # Backward-compatible scalar receipts remain valid only when they
                # prove exactly one entity for the reference's requested query.
                query = arguments.get("query")
                if (
                    meta.get("ambiguous")
                    or meta.get("requires_clarification")
                    or len(entity_ids) != 1
                    or (
                        not supplied_ids
                        and query not in {reference.canonical_name, reference.mention}
                    )
                ):
                    continue
                entity_id = next(iter(entity_ids))
                if supplied_ids and entity_id not in supplied_ids:
                    continue
            previous = bindings.get(index)
            if previous is not None and previous != entity_id:
                continue
            bindings[index] = entity_id
            matched.append(index)
        return matched

    def _entity_receipts_satisfy_task(
        self,
        task: ResearchTask,
        receipts: list[dict[str, Any]],
        bindings: dict[int, str],
        records_by_id: dict[str, dict[str, Any]],
    ) -> bool:
        ids = {
            str(item)
            for receipt in receipts
            for item in receipt.get("record_ids", [])
            if str(item) in records_by_id
        }
        indexes = self._task_reference_indexes(task)
        # Name and verified-ID phases may complete across several bounded calls,
        # but every reference still needs a current successful entity receipt.
        # A pre-existing context binding alone is not tool evidence for this task.
        return not indexes or all(bindings.get(index) in ids for index in indexes)

    def _receipt_matches_relation_task(
        self,
        task: ResearchTask,
        receipt: dict[str, Any],
        bindings: dict[int, str],
        goal_results: dict[str, _GoalResult],
    ) -> bool:
        expected = self._relation_arguments(task, bindings, goal_results)
        if expected is None:
            return False
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
        goal_results: dict[str, _GoalResult],
    ) -> list[_TaskContract]:
        raw: list[_TaskContract] = []
        for task in planner.research_tasks:
            if task.task_id in completed or not set(task.depends_on) <= completed:
                continue
            indexes = self._task_reference_indexes(task)
            if task.tool is ToolName.RELATIONS:
                if (
                    not set(indexes) <= bindings.keys()
                    or not self._task_dynamic_inputs_ready(task, goal_results)
                ):
                    continue
                arguments = self._relation_arguments(task, bindings, goal_results)
                if arguments is None:
                    continue
                raw.append(
                    _TaskContract(
                        task_ids=(task.task_id,),
                        tool=task.tool,
                        reference_indexes=tuple(indexes),
                        arguments=arguments,
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
                    match_mode = self._next_entity_match_mode(
                        state,
                        task.tool,
                        index,
                        reference.mention,
                        reference.canonical_name,
                    )
                    if match_mode is None:
                        continue
                    rewrite = (
                        (
                            reference.mention,
                            str(reference.canonical_name),
                        ),
                    ) if match_mode is MatchMode.CROSS_LANGUAGE_EXACT else ()
                    candidate = (
                        ()
                        if rewrite
                        else (reference.mention,)
                    )
                    raw.append(
                        _TaskContract(
                            task_ids=(task.task_id,),
                            tool=task.tool,
                            reference_indexes=(index,),
                            candidate_queries=candidate,
                            query_rewrites=rewrite,
                            reference_queries=((index, reference.mention),),
                            requested_attributes=tuple(task.requested_attributes),
                            required_match_mode=match_mode,
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
                        required_match_mode=MatchMode.EXACT,
                    )
                )

        grouped: dict[str, _TaskContract] = {}
        for contract in raw:
            # Entity resolution contracts with the same tool/mode/projection can
            # be executed in one native call.  Reference/query alignment remains
            # explicit in the contract because result records themselves are
            # deduplicated by stable entity ID.
            batch_entity = (
                contract.tool in {ToolName.PERSONS, ToolName.COMPANIES}
                and contract.arguments is None
            )
            identity = json.dumps(
                {
                    "tool": contract.tool.value,
                    "entity_kind": (
                        "query"
                        if contract.candidate_queries
                        else "ids"
                        if contract.allowed_entity_ids
                        else None
                    ),
                    "references": () if batch_entity else contract.reference_indexes,
                    "arguments": contract.arguments,
                    "queries": () if batch_entity else contract.candidate_queries,
                    "query_rewrites": (
                        () if batch_entity else contract.query_rewrites
                    ),
                    "ids": () if batch_entity else contract.allowed_entity_ids,
                    "attrs": contract.requested_attributes,
                    "match_mode": (
                        contract.required_match_mode.value
                        if contract.required_match_mode is not None
                        else None
                    ),
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
                    reference_indexes=tuple(
                        dict.fromkeys(
                            [*previous.reference_indexes, *contract.reference_indexes]
                        )
                    ),
                    arguments=contract.arguments,
                    candidate_queries=tuple(
                        dict.fromkeys(
                            [*previous.candidate_queries, *contract.candidate_queries]
                        )
                    ),
                    query_rewrites=tuple(
                        dict.fromkeys(
                            [*previous.query_rewrites, *contract.query_rewrites]
                        )
                    ),
                    reference_queries=tuple(
                        dict.fromkeys(
                            [*previous.reference_queries, *contract.reference_queries]
                        )
                    ),
                    allowed_entity_ids=tuple(
                        sorted(
                            {
                                *previous.allowed_entity_ids,
                                *contract.allowed_entity_ids,
                            }
                        )
                    ),
                    requested_attributes=contract.requested_attributes,
                    required_match_mode=contract.required_match_mode,
                )
        return list(grouped.values())

    @staticmethod
    def _next_entity_match_mode(
        state: AgentState,
        tool: ToolName,
        reference_index: int,
        mention: str,
        canonical_name: str | None,
    ) -> MatchMode | None:
        """Project mention exact -> mention fuzzy -> audited canonical exact."""

        attempts: list[tuple[MatchMode, dict[str, Any]]] = []
        for receipt in state.get("research_transcript", []):
            if (
                receipt.get("tool") != tool.value
                or not receipt.get("executed", True)
                or not receipt.get("success")
            ):
                continue
            arguments = receipt.get("arguments") or {}
            requested_queries = [
                *(
                    [str(arguments.get("query"))]
                    if arguments.get("query") is not None
                    else []
                ),
                *[str(item) for item in arguments.get("queries", [])],
            ]
            rewrites = [
                item
                for item in arguments.get("query_rewrites", [])
                if isinstance(item, dict)
            ]
            try:
                mode = MatchMode(arguments.get("match_mode", MatchMode.EXACT.value))
            except ValueError:
                continue
            if mode is MatchMode.CROSS_LANGUAGE_EXACT:
                if canonical_name is None or not any(
                    normalize_query(str(item.get("original_query", "")))
                    == normalize_query(mention)
                    and normalize_query(str(item.get("rewritten_query", "")))
                    == normalize_query(canonical_name)
                    for item in rewrites
                ):
                    continue
            elif normalize_query(mention) not in {
                normalize_query(item) for item in requested_queries
            }:
                continue
            indexes = receipt.get("reference_indexes", [])
            if indexes and reference_index not in indexes:
                continue
            meta = receipt.get("meta") or {}
            per_query = next(
                (
                    item
                    for item in meta.get("query_matches", [])
                    if isinstance(item, dict)
                    and normalize_query(str(item.get("query", "")))
                    == normalize_query(mention)
                ),
                None,
            )
            attempts.append((mode, per_query or meta))

        if not attempts:
            return MatchMode.EXACT
        if any(
            mode is MatchMode.CROSS_LANGUAGE_EXACT for mode, _meta in attempts
        ):
            return None
        exact_meta = [meta for mode, meta in attempts if mode is MatchMode.EXACT]
        if not exact_meta:
            return MatchMode.EXACT
        if not Researcher._complete_empty_entity_outcome(exact_meta[-1]):
            return None
        fuzzy_meta = [meta for mode, meta in attempts if mode is MatchMode.FUZZY]
        if not fuzzy_meta:
            return MatchMode.FUZZY
        if not Researcher._complete_empty_entity_outcome(fuzzy_meta[-1]):
            return None
        if (
            canonical_name is not None
            and normalize_query(canonical_name) != normalize_query(mention)
        ):
            return MatchMode.CROSS_LANGUAGE_EXACT
        return None

    @staticmethod
    def _complete_empty_entity_outcome(meta: dict[str, Any]) -> bool:
        return (
            int(meta.get("returned", 0)) == 0
            and not bool(meta.get("truncated"))
            and not bool(meta.get("ambiguous"))
        )

    @staticmethod
    def _task_dynamic_inputs_ready(
        task: ResearchTask, goal_results: dict[str, _GoalResult]
    ) -> bool:
        required = {*task.subject_result_goal_ids, *task.object_result_goal_ids}
        return all(
            goal_id in goal_results and goal_results[goal_id].complete
            for goal_id in required
        )

    @staticmethod
    def _control_fallback_not_required(
        task: ResearchTask,
        planner: PlannerDecision,
        completed: set[str],
        task_receipts: dict[str, list[dict[str, Any]]],
        records_by_id: dict[str, dict[str, Any]],
    ) -> bool:
        if task.tool is not ToolName.RELATIONS or task.goal_id is None:
            return False
        goals = {goal.goal_id: goal for goal in planner.research_goals}
        goal = goals.get(task.goal_id)
        if (
            goal is None
            or goal.intent is not Intent.FIND_CONTROLLED_COMPANIES
            or set(task.relation_types) == {RelationType.CONTROLS}
        ):
            return False
        explicit_tasks = [
            candidate
            for candidate in planner.research_tasks
            if candidate.goal_id == task.goal_id
            and candidate.tool is ToolName.RELATIONS
            and set(candidate.relation_types) == {RelationType.CONTROLS}
        ]
        if not explicit_tasks or any(
            candidate.task_id not in completed for candidate in explicit_tasks
        ):
            return False
        return any(
            records_by_id.get(str(record_id), {}).get("record_kind") == "relation"
            for candidate in explicit_tasks
            for receipt in task_receipts.get(candidate.task_id, [])
            for record_id in receipt.get("record_ids", [])
        )

    @classmethod
    def _task_input_ids(
        cls,
        task: ResearchTask,
        bindings: dict[int, str],
        goal_results: dict[str, _GoalResult],
    ) -> tuple[set[str], set[str]] | None:
        if not cls._task_dynamic_inputs_ready(task, goal_results):
            return None
        subjects = {
            bindings[index]
            for index in task.subject_reference_indexes
            if index in bindings
        }
        objects = {
            bindings[index]
            for index in task.object_reference_indexes
            if index in bindings
        }
        for goal_id in task.subject_result_goal_ids:
            subjects.update(goal_results[goal_id].result_entity_ids)
        for goal_id in task.object_result_goal_ids:
            objects.update(goal_results[goal_id].result_entity_ids)
        return subjects, objects

    @classmethod
    def _task_has_verified_empty_input(
        cls,
        task: ResearchTask,
        goal_results: dict[str, _GoalResult],
        bindings: dict[int, str],
    ) -> bool:
        if not (task.subject_result_goal_ids or task.object_result_goal_ids):
            return False
        inputs = cls._task_input_ids(task, bindings, goal_results)
        if inputs is None:
            return False
        subjects, objects = inputs
        return not subjects or (bool(task.object_result_goal_ids) and not objects)

    @classmethod
    def _relation_arguments(
        cls,
        task: ResearchTask,
        bindings: dict[int, str],
        goal_results: dict[str, _GoalResult],
    ) -> dict[str, Any] | None:
        inputs = cls._task_input_ids(task, bindings, goal_results)
        if inputs is None:
            return None
        subjects, objects = inputs
        if not subjects or (task.object_result_goal_ids and not objects):
            return None
        direction = {
            ResearchDirection.OUTGOING: RelationDirection.OUTGOING.value,
            ResearchDirection.INCOMING: RelationDirection.INCOMING.value,
            ResearchDirection.ANY: RelationDirection.ANY.value,
        }[task.direction]
        return {
            "subject_ids": sorted(subjects),
            "object_ids": sorted(objects),
            "relation_types": sorted(
                {item.value for item in task.relation_types}
            ),
            "raw_relation_types": sorted(set(task.raw_relation_types)),
            "direction": direction,
            "include_endpoints": True,
            "limit": 200,
        }

    def _build_goal_results(
        self,
        planner: PlannerDecision,
        completed: set[str],
        task_receipts: dict[str, list[dict[str, Any]]],
        records_by_id: dict[str, dict[str, Any]],
        bindings: dict[int, str],
    ) -> dict[str, _GoalResult]:
        results: dict[str, _GoalResult] = {}
        pending = list(planner.research_goals)
        while pending:
            progressed = False
            for goal in list(pending):
                if any(
                    dependency not in results or not results[dependency].complete
                    for dependency in goal.depends_on_goal_ids
                ):
                    continue
                goal_tasks = self._tasks_for_goal(planner, goal)
                complete = bool(goal_tasks) and all(
                    task.task_id in completed for task in goal_tasks
                )
                subject_ids = {
                    bindings[index]
                    for index in goal.subject_reference_indexes
                    if index in bindings
                }
                object_ids = {
                    bindings[index]
                    for index in goal.object_reference_indexes
                    if index in bindings
                }
                for dependency in goal.subject_result_goal_ids:
                    subject_ids.update(results[dependency].result_entity_ids)
                for dependency in goal.object_result_goal_ids:
                    object_ids.update(results[dependency].result_entity_ids)

                skipped_empty = bool(goal_tasks) and any(
                    task.task_id in completed
                    and not task_receipts.get(task.task_id)
                    and bool(
                        task.subject_result_goal_ids
                        or task.object_result_goal_ids
                    )
                    for task in goal_tasks
                    if task.tool is ToolName.RELATIONS
                )
                selected: list[dict[str, Any]] = []
                result_ids: set[str] = set()
                focus_ids: set[str] = set()
                nonempty = False
                if complete:
                    relation_tasks = [
                        task for task in goal_tasks if task.tool is ToolName.RELATIONS
                    ]
                    if relation_tasks:
                        selected, result_ids = self._project_goal_relations(
                            goal,
                            relation_tasks,
                            task_receipts,
                            records_by_id,
                            subject_ids,
                            object_ids,
                        )
                        nonempty = bool(selected)
                        if goal.intent is Intent.LOCATE_ENTITIES:
                            focus_ids = set(subject_ids)
                        elif goal.aggregation is ResultMergeStrategy.DIRECT:
                            focus_ids = {*subject_ids, *object_ids}
                        elif result_ids:
                            focus_ids = set(result_ids)
                        else:
                            focus_ids = {*subject_ids, *object_ids}
                        if nonempty and (
                            goal.intent is Intent.LOCATE_ENTITIES
                            or goal.aggregation is ResultMergeStrategy.DIRECT
                        ):
                            # These result shapes can legitimately retain focus
                            # entities with no incident selected edge: an isolated
                            # operand in an induced direct subgraph, or a queried
                            # company with no headquarters row. Keep those already
                            # verified entity receipts in this goal's own proof set
                            # so per-goal validation never borrows evidence from an
                            # unrelated goal or the aggregate selection. This adds no
                            # edge and manufactures no fact.
                            selected.extend(
                                records_by_id[entity_id]
                                for entity_id in sorted(focus_ids)
                                if entity_id in records_by_id
                                and records_by_id[entity_id].get("record_kind")
                                == "entity"
                            )
                    else:
                        entity_ids = {
                            bindings[index]
                            for index in {
                                *goal.subject_reference_indexes,
                                *goal.object_reference_indexes,
                            }
                            if index in bindings
                        }
                        selected = [
                            records_by_id[entity_id]
                            for entity_id in sorted(entity_ids)
                            if entity_id in records_by_id
                            and records_by_id[entity_id].get("record_kind") == "entity"
                        ]
                        result_ids = set(entity_ids)
                        focus_ids = set(entity_ids)
                        nonempty = bool(selected)
                if skipped_empty and not nonempty:
                    focus_ids = set()
                results[goal.goal_id] = _GoalResult(
                    goal=goal,
                    complete=complete,
                    skipped_empty_input=skipped_empty,
                    subject_ids=subject_ids,
                    explicit_object_ids=object_ids,
                    result_entity_ids=result_ids,
                    focus_entity_ids=focus_ids,
                    selected_records=_deduplicate_records(selected),
                    nonempty=nonempty,
                )
                pending.remove(goal)
                progressed = True
            if not progressed:
                break
        return results

    @staticmethod
    def _tasks_for_goal(
        planner: PlannerDecision, goal: ResearchGoal
    ) -> list[ResearchTask]:
        tasks = [
            task for task in planner.research_tasks if task.goal_id == goal.goal_id
        ]
        profile_tool = {
            Intent.GET_PERSON_PROFILE: ToolName.PERSONS,
            Intent.GET_COMPANY_PROFILE: ToolName.COMPANIES,
        }.get(goal.intent)
        if profile_tool is None:
            return tasks

        goal_indexes = {
            *goal.subject_reference_indexes,
            *goal.object_reference_indexes,
        }
        for task in planner.research_tasks:
            if task.goal_id is not None or task.tool is not profile_tool:
                continue
            task_indexes = set(Researcher._task_reference_indexes(task))
            if goal_indexes <= task_indexes and set(goal.requested_attributes) <= set(
                task.requested_attributes
            ):
                tasks.append(task)
        return list({task.task_id: task for task in tasks}.values())

    def _project_goal_relations(
        self,
        goal: ResearchGoal,
        tasks: list[ResearchTask],
        task_receipts: dict[str, list[dict[str, Any]]],
        records_by_id: dict[str, dict[str, Any]],
        subject_ids: set[str],
        explicit_object_ids: set[str],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        active_tasks = list(tasks)
        if goal.intent is Intent.FIND_CONTROLLED_COMPANIES:
            explicit = [
                task
                for task in tasks
                if set(task.relation_types) == {RelationType.CONTROLS}
            ]
            explicit_records = self._relation_records_for_tasks(
                explicit, task_receipts, records_by_id
            )
            active_tasks = explicit if explicit_records else [
                task for task in tasks if task not in explicit
            ]
        relations = self._relation_records_for_tasks(
            active_tasks, task_receipts, records_by_id
        )
        if not relations or not subject_ids:
            return [], set()

        if goal.aggregation is ResultMergeStrategy.DIRECT:
            signed = {*subject_ids, *explicit_object_ids}
            selected = [
                record
                for record in relations
                if str(record.get("source")) in signed
                and str(record.get("target")) in signed
            ]
            endpoints = {
                str(endpoint)
                for record in selected
                for endpoint in (record.get("source"), record.get("target"))
                if endpoint is not None
            }
            return _deduplicate_records(selected), endpoints

        allowed_types = {item.value for item in goal.target_types}
        memberships: dict[str, set[str]] = {subject: set() for subject in subject_ids}
        relation_memberships: dict[str, set[tuple[str, str]]] = {}
        for record in relations:
            pairs: set[tuple[str, str]] = set()
            source = str(record.get("source", ""))
            target = str(record.get("target", ""))
            for subject in subject_ids:
                neighbour: str | None = None
                if goal.direction is ResearchDirection.OUTGOING and source == subject:
                    neighbour = target
                elif goal.direction is ResearchDirection.INCOMING and target == subject:
                    neighbour = source
                elif goal.direction is ResearchDirection.ANY:
                    if source == subject:
                        neighbour = target
                    elif target == subject:
                        neighbour = source
                if not neighbour:
                    continue
                if explicit_object_ids and neighbour not in explicit_object_ids:
                    continue
                if allowed_types and str(
                    records_by_id.get(neighbour, {}).get("entity_type", "")
                ) not in allowed_types:
                    continue
                memberships[subject].add(neighbour)
                pairs.add((subject, neighbour))
            if pairs:
                relation_memberships[str(record.get("id"))] = pairs

        if goal.aggregation is ResultMergeStrategy.INTERSECTION:
            result_ids = (
                set.intersection(*memberships.values()) if memberships else set()
            )
        else:
            result_ids = set().union(*memberships.values()) if memberships else set()
        selected = [
            record
            for record in relations
            if any(
                neighbour in result_ids
                for _subject, neighbour in relation_memberships.get(
                    str(record.get("id")), set()
                )
            )
        ]
        return _deduplicate_records(selected), result_ids

    @staticmethod
    def _relation_records_for_tasks(
        tasks: list[ResearchTask],
        task_receipts: dict[str, list[dict[str, Any]]],
        records_by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _deduplicate_records(
            [
                records_by_id[str(record_id)]
                for task in tasks
                for receipt in task_receipts.get(task.task_id, [])
                for record_id in receipt.get("record_ids", [])
                if str(record_id) in records_by_id
                and records_by_id[str(record_id)].get("record_kind") == "relation"
            ]
        )

    @staticmethod
    def _project_goal_results(
        planner: PlannerDecision,
        goal_results: dict[str, _GoalResult],
        records_by_id: dict[str, dict[str, Any]],
        bindings: dict[int, str],
    ) -> tuple[list[dict[str, Any]], bool]:
        result_records = [
            record
            for goal in planner.research_goals
            for record in goal_results.get(
                goal.goal_id,
                _GoalResult(goal, False, False, set(), set(), set(), set(), [], False),
            ).selected_records
        ]
        relation_endpoints = {
            str(endpoint)
            for record in result_records
            if record.get("record_kind") == "relation"
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        selected_ids = {
            *bindings.values(),
            *relation_endpoints,
            *(
                str(record.get("id"))
                for record in result_records
                if record.get("id") is not None
            ),
        }
        selected = [
            record
            for record_id, record in records_by_id.items()
            if record_id in selected_ids
        ]
        return _deduplicate_records(selected), any(
            result.complete and result.nonempty for result in goal_results.values()
        )

    def _finish(
        self, state: AgentState, decision: ResearcherDecision
    ) -> dict[str, Any]:
        completion_state = self._with_verified_context_records(state)
        snapshot = self._snapshot(completion_state)
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
        evidence_by_id = {
            item.id: item for item in completion_state.get("tool_evidence", [])
        }
        referenced_evidence = {
            str(evidence_id)
            for record in snapshot.selected_records
            for evidence_id in record.get("evidence_ids", [])
        }
        if referenced_evidence - evidence_by_id.keys():
            raise ValueError("selected records reference missing Evidence")

        signature = self._derive_signature(completion_state, snapshot)
        if snapshot.result_nonempty:
            validate_signature_records(
                signature,
                snapshot.selected_records,
                completion_state.get("research_records", []),
            )
        query_resolved = {
            reference.mention: snapshot.bindings[index]
            for index, reference in enumerate(snapshot.planner.entity_references)
            if reference.source is EntityReferenceSource.CURRENT_QUERY
            and index in snapshot.bindings
        }
        if snapshot.planner.research_goals:
            focus = expected_focus_entity_ids(
                signature,
                snapshot.selected_records,
                completion_state.get("research_records", []),
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
            "research_records": completion_state.get("research_records", []),
            "tool_evidence": completion_state.get("tool_evidence", []),
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

    @staticmethod
    def _with_verified_context_records(state: AgentState) -> dict[str, Any]:
        """Materialize only Planner-approved prior graph entities for completion.

        Context bindings are stable IDs from the last verified focus.  A batched
        relation lookup may return no edge for one member, so the current tool
        receipt cannot reproduce that entity record.  Reusing the exact node and
        Evidence already present in the validated session graph closes the current
        graph without treating assistant prose, aliases, or arbitrary session IDs
        as facts.
        """

        try:
            planner = PlannerDecision.model_validate(state.get("planner_decision"))
        except (ValidationError, TypeError, ValueError):
            return dict(state)
        context_ids = {
            str(reference.context_entity_id)
            for reference in planner.entity_references
            if reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
            and reference.context_entity_id is not None
        }
        if not context_ids:
            return dict(state)
        try:
            graph = GraphPayload.model_validate(state.get("session_graph"))
        except (ValidationError, TypeError, ValueError):
            return dict(state)

        nodes_by_id = {node.id: node for node in graph.nodes}
        evidence_by_id = {item.id: item for item in graph.evidence}
        reusable_records: list[dict[str, Any]] = []
        reusable_evidence: list[Evidence] = []
        for entity_id in sorted(context_ids):
            node = nodes_by_id.get(entity_id)
            if node is None or not node.evidence_ids:
                continue
            if set(node.evidence_ids) - evidence_by_id.keys():
                continue
            reusable_records.append(
                {
                    "id": node.id,
                    "record_kind": "entity",
                    "entity_type": node.type.value,
                    "label": node.label,
                    "properties": dict(node.properties),
                    "evidence_ids": list(node.evidence_ids),
                }
            )
            reusable_evidence.extend(
                evidence_by_id[evidence_id] for evidence_id in node.evidence_ids
            )

        updated = dict(state)
        updated["research_records"] = _deduplicate_records(
            [*state.get("research_records", []), *reusable_records]
        )
        updated["tool_evidence"] = _deduplicate_evidence(
            [
                *(
                    item
                    if isinstance(item, Evidence)
                    else Evidence.model_validate(item)
                    for item in state.get("tool_evidence", [])
                ),
                *reusable_evidence,
            ]
        )
        return updated

    def _derive_signature(
        self, state: AgentState, snapshot: _TaskSnapshot
    ) -> QuerySignature:
        planner = snapshot.planner
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

        context_ids = {
            str(reference.context_entity_id)
            for reference in planner.entity_references
            if reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
            and reference.context_entity_id is not None
        }
        goal_signatures: list[QueryGoalSignature] = []
        for goal in planner.research_goals:
            result = snapshot.goal_results[goal.goal_id]
            goal_tasks = [
                task
                for task in planner.research_tasks
                if task.goal_id == goal.goal_id
                and task.task_id in snapshot.completed
            ]
            relation_tasks = [
                task for task in goal_tasks if task.tool is ToolName.RELATIONS
            ]
            if goal.intent is Intent.FIND_CONTROLLED_COMPANIES:
                explicit_tasks = [
                    task
                    for task in relation_tasks
                    if set(task.relation_types) == {RelationType.CONTROLS}
                ]
                if any(task_has_relations(task) for task in explicit_tasks):
                    relation_tasks = explicit_tasks
            nonempty_tasks = [task for task in relation_tasks if task_has_relations(task)]
            effective_types = {
                relation_type
                for task in nonempty_tasks
                for relation_type in task.relation_types
            }
            if any(
                not task.relation_types and not task.raw_relation_types
                for task in nonempty_tasks
            ):
                effective_types = set()
            empty_types = {
                relation_type
                for task in relation_tasks
                if not task_has_relations(task)
                for relation_type in task.relation_types
            }
            result_record_ids = sorted(
                str(record["id"])
                for record in result.selected_records
                if record.get("id") is not None
            )
            result_status = (
                GoalResultStatus.NONEMPTY
                if result.nonempty
                else GoalResultStatus.SKIPPED_EMPTY_INPUT
                if result.skipped_empty_input
                else GoalResultStatus.VERIFIED_EMPTY
            )
            goal_context_ids = {
                str(planner.entity_references[index].context_entity_id)
                for index in {
                    *goal.subject_reference_indexes,
                    *goal.object_reference_indexes,
                }
                if index < len(planner.entity_references)
                and planner.entity_references[index].source
                is EntityReferenceSource.CONVERSATION_CONTEXT
                and planner.entity_references[index].context_entity_id is not None
            }
            goal_signatures.append(
                QueryGoalSignature(
                    goal_id=goal.goal_id,
                    intent=goal.intent,
                    subject_ids=sorted(result.subject_ids),
                    object_ids=sorted(
                        {
                            *result.result_entity_ids,
                            *(
                                result.explicit_object_ids
                                if goal.aggregation is ResultMergeStrategy.DIRECT
                                or not result.nonempty
                                else set()
                            ),
                        }
                    ),
                    relation_types=sorted(effective_types, key=lambda item: item.value),
                    requested_relation_types=sorted(
                        {
                            relation_type
                            for task in relation_tasks
                            for relation_type in task.relation_types
                        }
                        or set(goal.relation_types),
                        key=lambda item: item.value,
                    ),
                    effective_relation_types=sorted(
                        effective_types, key=lambda item: item.value
                    ),
                    raw_relation_qualifiers=sorted(
                        {
                            item
                            for task in relation_tasks
                            for item in task.raw_relation_types
                        }
                    ),
                    verified_empty_relation_types=sorted(
                        empty_types, key=lambda item: item.value
                    ),
                    target_types=goal.target_types,
                    requested_attributes=goal.requested_attributes,
                    aggregation=goal.aggregation,
                    control_policy=goal.control_policy,
                    depends_on_goal_ids=goal.depends_on_goal_ids,
                    context_entity_ids=sorted(goal_context_ids),
                    result_status=result_status,
                    result_record_ids=result_record_ids,
                    focus_entity_ids=sorted(result.focus_entity_ids),
                )
            )

        subject_ids = {item for goal in goal_signatures for item in goal.subject_ids}
        object_ids = {item for goal in goal_signatures for item in goal.object_ids}
        requested_relation_types = {
            item for goal in goal_signatures for item in goal.requested_relation_types
        }
        effective_relation_types = {
            item for goal in goal_signatures for item in goal.effective_relation_types
        }
        raw_qualifiers = {
            item for goal in goal_signatures for item in goal.raw_relation_qualifiers
        }
        empty_types = {
            item
            for goal in goal_signatures
            for item in goal.verified_empty_relation_types
        }
        single_goal = goal_signatures[0] if len(goal_signatures) == 1 else None
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
                    node_type
                    for goal in goal_signatures
                    for node_type in goal.target_types
                )
            ),
            requested_attributes=list(
                dict.fromkeys(
                    attribute for goal in goal_signatures for attribute in goal.requested_attributes
                )
            ),
            context_entity_ids=sorted(context_ids),
            result_merge=planner.result_merge,
            control_policy=(
                single_goal.control_policy
                if single_goal is not None
                else ControlQueryPolicy.NOT_APPLICABLE
            ),
            entity_match_version=ENTITY_MATCH_ALGORITHM_VERSION,
            locale=str(state.get("locale", "zh-CN")),
            goals=goal_signatures,
        )

    def _payload(self, state: AgentState) -> dict[str, Any]:
        snapshot = self._snapshot(state)
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "plan": {
                "intent": snapshot.planner.intent.value,
                "result_merge": snapshot.planner.result_merge.value,
                "research_goals": [
                    goal.model_dump(mode="json")
                    for goal in snapshot.planner.research_goals
                ],
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
            "goal_status": [
                {
                    "goal_id": goal.goal_id,
                    "complete": snapshot.goal_results.get(goal.goal_id) is not None
                    and snapshot.goal_results[goal.goal_id].complete,
                    "result_nonempty": snapshot.goal_results.get(goal.goal_id)
                    is not None
                    and snapshot.goal_results[goal.goal_id].nonempty,
                    "skipped_empty_input": snapshot.goal_results.get(goal.goal_id)
                    is not None
                    and snapshot.goal_results[goal.goal_id].skipped_empty_input,
                }
                for goal in snapshot.planner.research_goals
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
                    "reference_indexes": list(
                        receipt.get("reference_indexes", [])
                    ),
                    "match_mode": (receipt.get("arguments") or {}).get(
                        "match_mode"
                    ),
                    "argument_fingerprint": receipt.get(
                        "argument_fingerprint"
                    ),
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
                    "research_failure_reason": (
                        "Researcher exhausted the absolute model-iteration limit"
                    ),
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
