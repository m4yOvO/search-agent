"""Two-stage prompt-driven Planner for the enterprise relationship graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agents.prompt_budget import (
    PromptBudgetExceeded,
    apply_prompt_budget,
)
from app.agents.prompts import (
    PLANNER_ANALYSIS_SYSTEM_PROMPT,
    build_planner_tasks_prompt,
)
from app.agents.state import AgentState
from app.llm import (
    ModelClient,
    ModelContractIssue,
    ModelInvocationError,
    ModelOutputContractError,
    safe_model_contract_issues,
)
from app.schemas import (
    EntityReference,
    EntityReferenceSource,
    GoalResultGrouping,
    Intent,
    NodeType,
    PlannerAnalysisDecision,
    PlannerContextSetKey,
    PlannerDecision,
    PlannerTaskDraft,
    PlannerTaskDecision,
    RelationType,
    ResearchDirection,
    ResearchGoal,
    ResearchTask,
    ResultMergeStrategy,
    TaskScopeSource,
    ToolName,
)


logger = logging.getLogger(__name__)

_CONTROL_FALLBACK_TYPES = {
    RelationType.FOUNDED,
    RelationType.WORKS_AT,
    RelationType.OWNS,
}
_CONTROL_FALLBACK_RAW = {
    "Founder_of",
    "Co-founder_of",
    "CEO_of",
    "Chairman_of",
    "Chairwoman_of",
    "Owns",
}

# Pydantic model-level validators attach their issue to the containing object.
# These stable Schema-code hints let the retry identify the exact field without
# retaining a rejected value, validation message, model payload, or query text.
_PLANNER_CONTRACT_FIELD_HINTS = {
    "clarification_question_required": "clarification_question",
    "clarification_question_forbidden": "clarification_question",
    "terminal_research_goals_must_be_empty": "research_goals",
    "entity_references_required": "entity_references",
    "research_goals_required": "research_goals",
    "research_goal_ids_must_be_unique": "research_goals[].goal_id",
    "goal_entity_index_out_of_range": "research_goals[].subject_reference_indexes",
    "goal_dependency_unknown": "research_goals[].depends_on_goal_ids",
    "entity_reference_not_used_by_goal": "research_goals[].subject_reference_indexes",
    "goal_dependency_cycle": "research_goals[].depends_on_goal_ids",
    "single_goal_intent_mismatch": "intent",
    "multi_goal_intent_required": "intent",
    "goal_self_dependency": "research_goals[].depends_on_goal_ids",
    "goal_subjects_required": "research_goals[].subject_reference_indexes",
    "goal_result_dependency_required": "research_goals[].depends_on_goal_ids",
    "goal_intent_not_executable": "research_goals[].intent",
    "control_policy_required": "research_goals[].control_policy",
    "control_scope_must_start_explicit": "research_goals[].relation_types",
    "control_policy_not_applicable": "research_goals[].control_policy",
    "location_relation_scope_required": "research_goals[].relation_types",
    "location_direction_must_be_outgoing": "research_goals[].direction",
    "location_target_type_required": "research_goals[].target_types",
    "profile_relation_scope_forbidden": "research_goals[].relation_types",
    "profile_attribute_not_supported": "research_goals[].requested_attributes",
    "relation_direction_required": "research_goals[].direction",
    "nary_aggregation_required": "research_goals[].aggregation",
    "location_aggregation_not_applicable": "research_goals[].aggregation",
    "direct_target_types_must_be_empty": "research_goals[].target_types",
    "consumed_goal_target_types_required": "research_goals[].target_types",
}
_PLANNER_RUNTIME_FIELD_HINTS = {
    "neighbor_goal_operand_overlap": (
        "research_goals[].object_reference_indexes"
    ),
}


class PlannerContractViolation(ValueError):
    """Typed Planner output failed one stable, non-factual runtime invariant."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def select_planner_task_profiles(
    analysis: PlannerAnalysisDecision,
) -> tuple[str, ...]:
    """Select at most two examples from typed semantics, never from query text."""

    goals = analysis.research_goals
    if len(goals) > 1:
        # ``multi_goal`` teaches the shared lookup/DAG shape. One additional
        # example may teach the most constrained goal-specific shape. This
        # selection is deliberately based only on typed ResearchGoal fields;
        # it never examines query text, entity labels, stable IDs, or fixture
        # cardinality.
        goal_profiles = {_task_profile_for_goal(goal) for goal in goals}
        for profile in (
            "control",
            "context_property",
            "nary_direct",
            "nary_intersection",
            "nary_union",
            "profile",
            "filtered_relation",
        ):
            if profile in goal_profiles:
                return ("multi_goal", profile)
        return ("multi_goal",)
    if not goals:
        return ("single_goal",)

    goal = goals[0]
    # A direct prior-focus location goal cannot be distinguished from a current
    # entity location goal using ResearchGoal alone. This single-goal fallback
    # is the only selector branch that consults typed reference provenance.
    if goal.intent is Intent.LOCATE_ENTITIES and any(
        analysis.entity_references[index].source
        is EntityReferenceSource.CONVERSATION_CONTEXT
        for index in goal.subject_reference_indexes
    ):
        return ("context_property",)
    profile = _task_profile_for_goal(goal)
    if profile != "single_goal":
        return (profile,)
    return ("single_goal",)


def _task_profile_for_goal(goal: ResearchGoal) -> str:
    """Map one typed goal to its task-example shape without user text."""

    if goal.intent is Intent.FIND_CONTROLLED_COMPANIES:
        return "control"
    if goal.intent in {Intent.GET_PERSON_PROFILE, Intent.GET_COMPANY_PROFILE}:
        return "profile"
    if goal.intent is Intent.LOCATE_ENTITIES and goal.subject_result_goal_ids:
        return "context_property"
    if goal.aggregation is ResultMergeStrategy.UNION:
        return "nary_union"
    if goal.aggregation is ResultMergeStrategy.INTERSECTION:
        return "nary_intersection"
    if goal.aggregation is ResultMergeStrategy.DIRECT:
        return "nary_direct"
    if goal.relation_types or goal.raw_relation_types:
        return "filtered_relation"
    return "single_goal"


@dataclass(slots=True)
class Planner:
    """Run semantic analysis and task-DAG planning as separate graph nodes."""

    model: ModelClient
    max_replans: int = 2
    max_research_steps: int = 8
    input_token_budget: int = 12_000
    entity_catalog: tuple[dict[str, str], ...] = ()
    raw_relation_vocabulary: tuple[str, ...] = ()
    available_tools: tuple[dict[str, str], ...] = ()

    async def analyze(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "planner_analyze"]
        is_new_replan = bool(state.get("needs_replan"))
        terminal_review_count = (
            0
            if is_new_replan
            else state.get("planner_terminal_review_count", 0)
        )
        retry_count = (
            0
            if is_new_replan
            else state.get("planner_analysis_retry_count", 0)
        )
        replan_count = state.get("replan_count", 0) + (1 if is_new_replan else 0)

        if replan_count > self.max_replans:
            return self._terminal_failure(
                state,
                route=route,
                stage="analysis",
                code="planner_replan_limit",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
            )

        payload = self._analysis_payload(state, replan_count=replan_count)
        try:
            budgeted = apply_prompt_budget(
                PLANNER_ANALYSIS_SYSTEM_PROMPT,
                payload,
                PlannerAnalysisDecision,
                self.input_token_budget,
            )
        except PromptBudgetExceeded:
            return self._terminal_failure(
                state,
                route=route,
                stage="analysis",
                code="planner_prompt_budget_exceeded",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
            )

        call_updates = self._model_call_updates(state)
        try:
            value = await self.model.structured(
                PLANNER_ANALYSIS_SYSTEM_PROMPT,
                budgeted.payload,
                PlannerAnalysisDecision,
                "planner_analysis",
            )
            analysis = PlannerAnalysisDecision.model_validate(value)
            self._validate_analysis(analysis, state)
        except ModelInvocationError:
            return self._retry_or_fail(
                state,
                route=route,
                stage="analysis",
                code="model_invocation_failed",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
                call_updates=call_updates,
            )
        except ModelOutputContractError as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="analysis",
                code="invalid_analysis_schema",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
                call_updates=call_updates,
                contract_issues=exc.issues,
            )
        except PlannerContractViolation as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="analysis",
                code=exc.code,
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
                call_updates=call_updates,
            )
        except ValidationError as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="analysis",
                code="invalid_analysis_schema",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
                call_updates=call_updates,
                contract_issues=safe_model_contract_issues(exc),
            )
        except (TypeError, ValueError):
            return self._retry_or_fail(
                state,
                route=route,
                stage="analysis",
                code="invalid_analysis_schema",
                retry_field="planner_analysis_retry_count",
                retry_count=retry_count,
                replan_count=replan_count,
                call_updates=call_updates,
            )

        terminal = analysis.intent in {Intent.CLARIFY, Intent.UNSUPPORTED} or (
            analysis.query_requires_realtime_data
        )
        requires_terminal_review = terminal and terminal_review_count == 0
        terminal_decision = (
            self._terminal_decision(analysis)
            if terminal and not requires_terminal_review
            else None
        )
        logger.info(
            "planner_analysis_accepted",
            extra={
                "event": "planner_analysis_accepted",
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id"),
                "intent": analysis.intent.value,
                "entity_reference_count": len(analysis.entity_references),
                "context_reference_count": sum(
                    reference.source is EntityReferenceSource.CONVERSATION_CONTEXT
                    for reference in analysis.entity_references
                ),
                "result_merge": analysis.result_merge.value,
                "estimated_prompt_tokens": budgeted.estimated_tokens,
                "trimmed_assistant_fields": budgeted.trimmed_assistant_fields,
                "trimmed_user_fields": budgeted.trimmed_user_fields,
                "terminal_review_pending": requires_terminal_review,
            },
        )
        update: dict[str, Any] = {
            **call_updates,
            "planner_analysis": None if requires_terminal_review else analysis,
            "planner_terminal_candidate": (
                analysis if requires_terminal_review else None
            ),
            "planner_terminal_review_pending": requires_terminal_review,
            "planner_terminal_review_count": (
                terminal_review_count + 1
                if requires_terminal_review
                else terminal_review_count
            ),
            "planner_task_plan": None,
            "planner_prompt_profile": [],
            "planner_contract_feedback": (
                {
                    "stage": "analysis",
                    "code": "terminal_semantic_review",
                }
                if requires_terminal_review
                else None
            ),
            "planner_analysis_retry_count": retry_count,
            "planner_task_retry_count": 0,
            "planner_decision": terminal_decision,
            "planner_failed": False,
            "needs_replan": False,
            "replan_count": replan_count,
            "research_failure_reason": None,
            "research_complete": False,
            "run_status": "running",
            "route_history": route,
        }
        if requires_terminal_review:
            update["agent_steps"] = [
                *state.get("agent_steps", []),
                self._safe_step("terminal_review_requested", None),
            ]
        elif terminal_decision is not None:
            update["agent_steps"] = [
                *state.get("agent_steps", []),
                self._safe_step("plan", None, terminal_decision),
            ]
        return update

    async def plan_tasks(self, state: AgentState) -> dict[str, Any]:
        route = [*state.get("route_history", []), "planner_tasks"]
        retry_count = state.get("planner_task_retry_count", 0)
        analysis_value = state.get("planner_analysis")
        try:
            analysis = PlannerAnalysisDecision.model_validate(analysis_value)
        except (ValidationError, TypeError, ValueError):
            return self._terminal_failure(
                state,
                route=route,
                stage="tasks",
                code="missing_validated_analysis",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
            )

        profiles = select_planner_task_profiles(analysis)
        try:
            system_prompt = build_planner_tasks_prompt(profiles)
            payload = self._tasks_payload(state, analysis, profiles)
            budgeted = apply_prompt_budget(
                system_prompt,
                payload,
                PlannerTaskDecision,
                self.input_token_budget,
            )
        except PromptBudgetExceeded:
            return self._terminal_failure(
                state,
                route=route,
                stage="tasks",
                code="planner_prompt_budget_exceeded",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
            )

        call_updates = self._model_call_updates(state)
        try:
            value = await self.model.structured(
                system_prompt,
                budgeted.payload,
                PlannerTaskDecision,
                "planner_tasks",
            )
            task_plan = PlannerTaskDecision.model_validate(value)
            task_plan = self._inherit_goal_operands(analysis, task_plan)
            self._validate_task_plan(analysis, task_plan)
            decision = self._assemble_decision(analysis, task_plan, state)
        except ModelInvocationError:
            return self._retry_or_fail(
                state,
                route=route,
                stage="tasks",
                code="model_invocation_failed",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
                call_updates=call_updates,
            )
        except ModelOutputContractError as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="tasks",
                code="invalid_task_schema",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
                call_updates=call_updates,
                contract_issues=exc.issues,
            )
        except PlannerContractViolation as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="tasks",
                code=exc.code,
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
                call_updates=call_updates,
            )
        except ValidationError as exc:
            return self._retry_or_fail(
                state,
                route=route,
                stage="tasks",
                code="invalid_task_schema",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
                call_updates=call_updates,
                contract_issues=safe_model_contract_issues(exc),
            )
        except (TypeError, ValueError):
            return self._retry_or_fail(
                state,
                route=route,
                stage="tasks",
                code="invalid_task_schema",
                retry_field="planner_task_retry_count",
                retry_count=retry_count,
                replan_count=state.get("replan_count", 0),
                call_updates=call_updates,
            )

        logger.info(
            "planner_tasks_accepted",
            extra={
                "event": "planner_tasks_accepted",
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id"),
                "intent": decision.intent.value,
                "research_task_count": len(decision.research_tasks),
                "entity_reference_count": len(decision.entity_references),
                "context_entity_count": sum(
                    reference.context_entity_id is not None
                    for reference in decision.entity_references
                ),
                "profiles": list(profiles),
                "estimated_prompt_tokens": budgeted.estimated_tokens,
            },
        )
        resolved = dict(state.get("resolved_entities", {}))
        return {
            **call_updates,
            "planner_task_plan": task_plan,
            "planner_prompt_profile": list(profiles),
            "planner_contract_feedback": None,
            "planner_decision": decision,
            "planner_failed": False,
            "resolved_entities": resolved,
            "needs_replan": False,
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

    def _validate_analysis(
        self, analysis: PlannerAnalysisDecision, state: AgentState
    ) -> None:
        query = str(state.get("current_query", ""))
        catalog_types: dict[str, set[NodeType]] = {}
        for item in self.entity_catalog:
            try:
                name = str(item["name"])
                node_type = NodeType(str(item["entity_type"]))
            except (KeyError, TypeError, ValueError):
                continue
            catalog_types.setdefault(name, set()).add(node_type)

        context_members = self._trusted_focus_members(state)
        for reference in analysis.entity_references:
            if reference.mention not in query:
                raise PlannerContractViolation("current_mention_absent")
            if reference.source is EntityReferenceSource.CURRENT_QUERY:
                if reference.canonical_name is not None:
                    matching_types = catalog_types.get(reference.canonical_name)
                    if catalog_types and not matching_types:
                        raise PlannerContractViolation("canonical_name_unknown")
                    if matching_types and not matching_types.intersection(
                        reference.expected_types
                    ):
                        raise PlannerContractViolation("canonical_name_wrong_type")
            else:
                compatible = [
                    member
                    for member in context_members
                    if NodeType(member["entity_type"]) in reference.expected_types
                ]
                if not compatible:
                    raise PlannerContractViolation("context_set_type_mismatch")

        requested_raw = {
            raw
            for goal in analysis.research_goals
            for raw in goal.raw_relation_types
        }
        if any(
            goal.aggregation is ResultMergeStrategy.DIRECT and goal.target_types
            for goal in analysis.research_goals
        ):
            raise PlannerContractViolation("direct_target_types_forbidden")
        if any(
            goal.aggregation is not ResultMergeStrategy.DIRECT
            and set(goal.subject_reference_indexes).intersection(
                goal.object_reference_indexes
            )
            for goal in analysis.research_goals
        ):
            raise PlannerContractViolation("neighbor_goal_operand_overlap")
        # Entity count must never be mistaken for goal count. Equivalent-scope
        # root goals normally represent an accidental split of one N-ary
        # objective. The model can, however, explicitly type each goal as a
        # separately addressable result group. This validation consumes only
        # typed semantics and never query text, entity names, IDs, or test data.
        parallel_scopes: dict[tuple[Any, ...], list[ResearchGoal]] = {}
        for goal in analysis.research_goals:
            if (
                goal.intent
                not in {
                    Intent.FIND_RELATED_COMPANIES,
                    Intent.FIND_CONTROLLED_COMPANIES,
                }
                or goal.object_reference_indexes
                or goal.subject_result_goal_ids
                or goal.object_result_goal_ids
                or goal.depends_on_goal_ids
                or goal.aggregation
                not in {
                    ResultMergeStrategy.NOT_APPLICABLE,
                    ResultMergeStrategy.UNION,
                }
            ):
                continue
            scope_key = (
                goal.intent,
                tuple(sorted(goal.relation_types, key=lambda item: item.value)),
                tuple(sorted(goal.raw_relation_types)),
                goal.direction,
                tuple(sorted(goal.target_types, key=lambda item: item.value)),
                tuple(sorted(goal.requested_attributes)),
                goal.control_policy,
            )
            parallel_scopes.setdefault(scope_key, []).append(goal)
        if any(
            len(goals) > 1
            and len(
                {
                    index
                    for goal in goals
                    for index in goal.subject_reference_indexes
                }
            )
            > 1
            and not all(
                goal.result_grouping is GoalResultGrouping.SEPARATE
                for goal in goals
            )
            for goals in parallel_scopes.values()
        ):
            raise PlannerContractViolation("parallel_scope_must_be_one_nary_goal")
        if self.raw_relation_vocabulary and requested_raw - set(
            self.raw_relation_vocabulary
        ):
            raise PlannerContractViolation("raw_relation_unknown")

    @staticmethod
    def _inherit_goal_operands(
        analysis: PlannerAnalysisDecision,
        task_plan: PlannerTaskDecision,
    ) -> PlannerTaskDecision:
        """Make the validated stage-one goal the relation-task source of truth.

        Stage two chooses the executable DAG (tool, goal binding and dependencies),
        but its relation operand fields merely transport semantics already fixed by
        the corresponding :class:`ResearchGoal`.  Requiring the model to reproduce
        those lists byte-for-byte made a valid N-ary analysis fail when a task draft
        abbreviated the operands.  This projection is independent of query text,
        entity names and operand count; unknown goal IDs remain untouched so the
        ordinary contract validator can reject them.
        """

        goals = {goal.goal_id: goal for goal in analysis.research_goals}
        normalized_tasks: list[PlannerTaskDraft] = []
        for task in task_plan.research_tasks:
            if task.tool is not ToolName.RELATIONS or task.goal_id not in goals:
                normalized_tasks.append(task)
                continue

            goal = goals[task.goal_id]
            subject_indexes = list(goal.subject_reference_indexes)
            object_indexes = list(goal.object_reference_indexes)
            if goal.aggregation is ResultMergeStrategy.DIRECT:
                operands = list(dict.fromkeys([*subject_indexes, *object_indexes]))
                subject_indexes = operands
                object_indexes = operands

            normalized_tasks.append(
                task.model_copy(
                    update={
                        "subject_reference_indexes": subject_indexes,
                        "object_reference_indexes": object_indexes,
                        "subject_result_goal_ids": list(
                            goal.subject_result_goal_ids
                        ),
                        "object_result_goal_ids": list(goal.object_result_goal_ids),
                    }
                )
            )

        return PlannerTaskDecision.model_validate(
            {"research_tasks": normalized_tasks}
        )

    def _validate_task_plan(
        self,
        analysis: PlannerAnalysisDecision,
        task_plan: PlannerTaskDecision,
    ) -> None:
        reference_count = len(analysis.entity_references)
        goals = {goal.goal_id: goal for goal in analysis.research_goals}
        lookup_tasks: dict[int, set[str]] = {
            index: set() for index in range(reference_count)
        }
        entity_tasks: list[PlannerTaskDraft] = []
        relation_tasks: list[PlannerTaskDraft] = []
        tasks_by_goal: dict[str, list[PlannerTaskDraft]] = {
            goal_id: [] for goal_id in goals
        }
        current_lookup_task_ids: dict[ToolName, set[str]] = {
            ToolName.PERSONS: set(),
            ToolName.COMPANIES: set(),
        }

        for task in task_plan.research_tasks:
            indexes = [
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            ]
            if any(index >= reference_count for index in indexes):
                raise PlannerContractViolation("task_reference_invalid")
            if task.goal_id is not None and task.goal_id not in goals:
                raise PlannerContractViolation("task_goal_unknown")
            result_goal_ids = {
                *task.subject_result_goal_ids,
                *task.object_result_goal_ids,
            }
            if result_goal_ids - goals.keys():
                raise PlannerContractViolation("task_result_goal_unknown")
            if task.goal_id is not None:
                tasks_by_goal[task.goal_id].append(task)

            if task.tool in {ToolName.PERSONS, ToolName.COMPANIES}:
                entity_tasks.append(task)
                if task.subject_result_goal_ids or task.object_result_goal_ids:
                    raise PlannerContractViolation("entity_task_result_set_forbidden")
                if not indexes:
                    raise PlannerContractViolation("empty_entity_lookup")
                for index in indexes:
                    reference = analysis.entity_references[index]
                    required_type = (
                        NodeType.PERSON
                        if task.tool is ToolName.PERSONS
                        else NodeType.COMPANY
                    )
                    if required_type not in reference.expected_types:
                        raise PlannerContractViolation("entity_task_wrong_type")
                    if (
                        reference.source
                        is EntityReferenceSource.CONVERSATION_CONTEXT
                        and not any(
                            goal.intent
                            in {
                                Intent.GET_PERSON_PROFILE,
                                Intent.GET_COMPANY_PROFILE,
                            }
                            and index
                            in {
                                *goal.subject_reference_indexes,
                                *goal.object_reference_indexes,
                            }
                            and task.goal_id in {None, goal.goal_id}
                            for goal in goals.values()
                        )
                    ):
                        raise PlannerContractViolation("context_entity_relookup")
                    lookup_tasks[index].add(task.task_id)
                    if reference.source is EntityReferenceSource.CURRENT_QUERY:
                        current_lookup_task_ids[task.tool].add(task.task_id)
            elif task.tool is ToolName.RELATIONS:
                relation_tasks.append(task)

        for index, reference in enumerate(analysis.entity_references):
            if (
                reference.source is EntityReferenceSource.CURRENT_QUERY
                and len(lookup_tasks[index]) != 1
            ):
                raise PlannerContractViolation(
                    "missing_entity_lookup"
                    if not lookup_tasks[index]
                    else "duplicate_entity_lookup"
                )
        if any(len(task_ids) > 1 for task_ids in current_lookup_task_ids.values()):
            raise PlannerContractViolation("entity_lookups_not_batched")

        tasks_by_id = {task.task_id: task for task in task_plan.research_tasks}

        def dependency_closure(task: PlannerTaskDraft) -> set[str]:
            closure: set[str] = set()
            pending = list(task.depends_on)
            while pending:
                dependency = pending.pop()
                if dependency in closure:
                    continue
                closure.add(dependency)
                pending.extend(tasks_by_id[dependency].depends_on)
            return closure

        def task_ids_for_goal(goal_id: str) -> set[str]:
            """Return every factual task that realizes one typed goal.

            Profile goals may intentionally share one goal-neutral, type-batched
            entity lookup.  Treat that shared lookup as the factual task for each
            covered profile goal so downstream goal dependencies cannot silently
            omit it.
            """

            goal = goals[goal_id]
            task_ids = {task.task_id for task in tasks_by_goal[goal_id]}
            if goal.intent not in {
                Intent.GET_PERSON_PROFILE,
                Intent.GET_COMPANY_PROFILE,
            }:
                return task_ids
            expected_tool = (
                ToolName.PERSONS
                if goal.intent is Intent.GET_PERSON_PROFILE
                else ToolName.COMPANIES
            )
            goal_indexes = {
                *goal.subject_reference_indexes,
                *goal.object_reference_indexes,
            }
            task_ids.update(
                task.task_id
                for task in entity_tasks
                if task.goal_id is None
                and task.tool is expected_tool
                and goal_indexes
                <= {
                    *task.subject_reference_indexes,
                    *task.object_reference_indexes,
                }
            )
            return task_ids

        for task in relation_tasks:
            goal = goals[task.goal_id or ""]
            if set(task.subject_result_goal_ids) != set(
                goal.subject_result_goal_ids
            ) or set(task.object_result_goal_ids) != set(
                goal.object_result_goal_ids
            ):
                raise PlannerContractViolation("task_result_scope_mismatch")
            goal_operands = list(
                dict.fromkeys(
                    [
                        *goal.subject_reference_indexes,
                        *goal.object_reference_indexes,
                    ]
                )
            )
            if goal.aggregation is ResultMergeStrategy.DIRECT:
                if (
                    task.subject_reference_indexes != goal_operands
                    or task.object_reference_indexes != goal_operands
                ):
                    raise PlannerContractViolation("direct_task_incomplete")
            elif (
                task.subject_reference_indexes != goal.subject_reference_indexes
                or task.object_reference_indexes != goal.object_reference_indexes
            ):
                raise PlannerContractViolation("task_operand_mismatch")

            referenced = set(goal_operands)
            transitive_dependencies = dependency_closure(task)
            for index in referenced:
                reference = analysis.entity_references[index]
                if reference.source is EntityReferenceSource.CURRENT_QUERY and not (
                    lookup_tasks[index] & transitive_dependencies
                ):
                    raise PlannerContractViolation("missing_lookup_dependency")
            for dependency_goal_id in goal.depends_on_goal_ids:
                dependency_task_ids = task_ids_for_goal(dependency_goal_id)
                if dependency_task_ids and not (
                    dependency_task_ids & transitive_dependencies
                ):
                    raise PlannerContractViolation("missing_goal_dependency")

        for goal in analysis.research_goals:
            goal_tasks = tasks_by_goal[goal.goal_id]
            goal_relations = [
                task for task in goal_tasks if task.tool is ToolName.RELATIONS
            ]
            if goal.intent in {
                Intent.GET_PERSON_PROFILE,
                Intent.GET_COMPANY_PROFILE,
            }:
                if goal_relations:
                    raise PlannerContractViolation("profile_relation_task_forbidden")
                profile_indexes = {
                    *goal.subject_reference_indexes,
                    *goal.object_reference_indexes,
                }
                matching_entity_tasks = [
                    task
                    for task in entity_tasks
                    if task.tool in {ToolName.PERSONS, ToolName.COMPANIES}
                    and task.goal_id in {None, goal.goal_id}
                    and profile_indexes
                    <= {
                        *task.subject_reference_indexes,
                        *task.object_reference_indexes,
                    }
                ]
                if len(matching_entity_tasks) != 1:
                    raise PlannerContractViolation("profile_task_incomplete")
                continue

            if goal.intent is Intent.FIND_CONTROLLED_COMPANIES:
                explicit = [
                    task
                    for task in goal_relations
                    if task.scope_source is TaskScopeSource.CONTROL_EXPLICIT
                ]
                fallback = [
                    task
                    for task in goal_relations
                    if task.scope_source is TaskScopeSource.CONTROL_FALLBACK
                ]
                if len(explicit) != 1 or len(fallback) != 1:
                    raise PlannerContractViolation("control_stages_incomplete")
                if explicit[0].task_id not in dependency_closure(fallback[0]):
                    raise PlannerContractViolation("control_dependency_invalid")
            elif (
                len(goal_relations) != 1
                or goal_relations[0].scope_source is not TaskScopeSource.GOAL
            ):
                raise PlannerContractViolation("goal_relation_task_incomplete")

    def _assemble_decision(
        self,
        analysis: PlannerAnalysisDecision,
        task_plan: PlannerTaskDecision,
        state: AgentState,
    ) -> PlannerDecision:
        canonical_goals, goal_id_map = self._canonicalize_goals(
            analysis.research_goals
        )
        canonical_drafts = [
            task.model_copy(
                update={
                    "goal_id": (
                        goal_id_map[task.goal_id]
                        if task.goal_id is not None
                        else None
                    ),
                    "subject_result_goal_ids": [
                        goal_id_map[item] for item in task.subject_result_goal_ids
                    ],
                    "object_result_goal_ids": [
                        goal_id_map[item] for item in task.object_result_goal_ids
                    ],
                }
            )
            for task in task_plan.research_tasks
        ]
        final_references: list[EntityReference] = []
        index_map: dict[int, list[int]] = {}
        trusted_members = self._trusted_focus_members(state)

        for old_index, reference in enumerate(analysis.entity_references):
            new_indexes: list[int] = []
            if reference.source is EntityReferenceSource.CURRENT_QUERY:
                new_indexes.append(len(final_references))
                final_references.append(
                    EntityReference(
                        mention=reference.mention,
                        source=reference.source,
                        role=reference.role,
                        expected_types=reference.expected_types,
                        canonical_name=reference.canonical_name,
                        context_entity_id=None,
                    )
                )
            else:
                compatible = [
                    member
                    for member in trusted_members
                    if NodeType(member["entity_type"]) in reference.expected_types
                ]
                if not compatible:
                    raise PlannerContractViolation("context_set_type_mismatch")
                for member in compatible:
                    new_indexes.append(len(final_references))
                    final_references.append(
                        EntityReference(
                            mention=reference.mention,
                            source=reference.source,
                            role=reference.role,
                            expected_types=reference.expected_types,
                            canonical_name=None,
                            context_entity_id=member["entity_id"],
                        )
                    )
            index_map[old_index] = new_indexes

        expanded_goals = [
            self._expand_goal(goal, index_map) for goal in canonical_goals
        ]
        goals_by_id = {goal.goal_id: goal for goal in expanded_goals}
        expanded_tasks = [
            self._compile_task(task, goals_by_id, index_map)
            for task in canonical_drafts
        ]
        aggregate_merge = (
            expanded_goals[0].aggregation
            if len(expanded_goals) == 1
            else ResultMergeStrategy.NOT_APPLICABLE
        )
        decision = PlannerDecision(
            intent=analysis.intent,
            entity_references=final_references,
            research_goals=expanded_goals,
            research_tasks=expanded_tasks,
            result_merge=aggregate_merge,
            clarification_question=analysis.clarification_question,
            query_requires_realtime_data=analysis.query_requires_realtime_data,
        )
        self._validate_context_ids(decision, state)
        self._validate_typed_references(decision)
        self._validate_research_tasks(decision)

        used_final_indexes = {
            index
            for task in decision.research_tasks
            for index in [
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            ]
        }
        if any(
            index not in used_final_indexes
            for indexes in index_map.values()
            for index in indexes
        ):
            raise PlannerContractViolation("expanded_context_member_unused")
        return decision

    @staticmethod
    def _canonicalize_goals(
        goals: list[ResearchGoal],
    ) -> tuple[list[ResearchGoal], dict[str, str]]:
        """Normalize model-authored goal IDs without changing the typed DAG."""

        order = {goal.goal_id: index for index, goal in enumerate(goals)}
        remaining = {goal.goal_id: goal for goal in goals}
        emitted: list[ResearchGoal] = []
        emitted_ids: set[str] = set()
        while remaining:
            ready = [
                goal
                for goal in remaining.values()
                if set(goal.depends_on_goal_ids) <= emitted_ids
            ]
            if not ready:
                raise PlannerContractViolation("goal_dependency_cycle")
            ready.sort(key=lambda goal: order[goal.goal_id])
            for goal in ready:
                emitted.append(goal)
                emitted_ids.add(goal.goal_id)
                remaining.pop(goal.goal_id)

        goal_id_map = {
            goal.goal_id: f"goal_{index}"
            for index, goal in enumerate(emitted, start=1)
        }
        normalized = [
            ResearchGoal.model_validate(
                goal.model_copy(
                    update={
                        "goal_id": goal_id_map[goal.goal_id],
                        "depends_on_goal_ids": [
                            goal_id_map[item] for item in goal.depends_on_goal_ids
                        ],
                        "subject_result_goal_ids": [
                            goal_id_map[item]
                            for item in goal.subject_result_goal_ids
                        ],
                        "object_result_goal_ids": [
                            goal_id_map[item]
                            for item in goal.object_result_goal_ids
                        ],
                    }
                )
            )
            for goal in emitted
        ]
        return normalized, goal_id_map

    @staticmethod
    def _expand_indexes(
        values: list[int], index_map: dict[int, list[int]]
    ) -> list[int]:
        return list(
            dict.fromkeys(
                new_index
                for old_index in values
                for new_index in index_map[old_index]
            )
        )

    def _expand_goal(
        self, goal: ResearchGoal, index_map: dict[int, list[int]]
    ) -> ResearchGoal:
        subjects = self._expand_indexes(goal.subject_reference_indexes, index_map)
        objects = self._expand_indexes(goal.object_reference_indexes, index_map)
        if goal.aggregation is ResultMergeStrategy.DIRECT:
            operands = list(dict.fromkeys([*subjects, *objects]))
            subjects = operands
            objects = operands
        return ResearchGoal.model_validate(
            goal.model_copy(
                update={
                    "subject_reference_indexes": subjects,
                    "object_reference_indexes": objects,
                }
            )
        )

    def _compile_task(
        self,
        draft: PlannerTaskDraft,
        goals: dict[str, ResearchGoal],
        index_map: dict[int, list[int]],
    ) -> ResearchTask:
        goal = goals.get(draft.goal_id or "")
        subject_indexes = self._expand_indexes(
            draft.subject_reference_indexes, index_map
        )
        object_indexes = self._expand_indexes(
            draft.object_reference_indexes, index_map
        )
        relation_types: list[RelationType] = []
        raw_relation_types: list[str] = []
        direction = ResearchDirection.NOT_APPLICABLE
        target_types: list[NodeType] = []
        requested_attributes: list[str] = []

        if draft.tool is ToolName.RELATIONS:
            if goal is None:
                raise PlannerContractViolation("task_goal_unknown")
            subject_indexes = list(goal.subject_reference_indexes)
            object_indexes = list(goal.object_reference_indexes)
            if draft.scope_source is TaskScopeSource.GOAL:
                relation_types = list(goal.relation_types)
                raw_relation_types = list(goal.raw_relation_types)
                direction = goal.direction
                target_types = list(goal.target_types)
                requested_attributes = list(goal.requested_attributes)
            elif draft.scope_source is TaskScopeSource.CONTROL_EXPLICIT:
                relation_types = [RelationType.CONTROLS]
                direction = ResearchDirection.OUTGOING
                target_types = [NodeType.COMPANY]
            elif draft.scope_source is TaskScopeSource.CONTROL_FALLBACK:
                relation_types = sorted(
                    _CONTROL_FALLBACK_TYPES, key=lambda item: item.value
                )
                raw_relation_types = sorted(_CONTROL_FALLBACK_RAW)
                direction = ResearchDirection.OUTGOING
                target_types = [NodeType.COMPANY]
            else:
                raise PlannerContractViolation("task_scope_missing")
        elif goal is not None and goal.intent in {
            Intent.GET_PERSON_PROFILE,
            Intent.GET_COMPANY_PROFILE,
        }:
            target_types = list(goal.target_types)
            requested_attributes = list(goal.requested_attributes)
        elif goal is None and draft.tool in {
            ToolName.PERSONS,
            ToolName.COMPANIES,
        }:
            task_indexes = {*subject_indexes, *object_indexes}
            expected_intent = (
                Intent.GET_PERSON_PROFILE
                if draft.tool is ToolName.PERSONS
                else Intent.GET_COMPANY_PROFILE
            )
            covered_profiles = [
                candidate
                for candidate in goals.values()
                if candidate.intent is expected_intent
                and {
                    *candidate.subject_reference_indexes,
                    *candidate.object_reference_indexes,
                }
                <= task_indexes
            ]
            target_types = list(
                dict.fromkeys(
                    item
                    for candidate in covered_profiles
                    for item in candidate.target_types
                )
            )
            requested_attributes = list(
                dict.fromkeys(
                    item
                    for candidate in covered_profiles
                    for item in candidate.requested_attributes
                )
            )

        return ResearchTask(
            task_id=draft.task_id,
            goal_id=draft.goal_id,
            subject_result_goal_ids=list(draft.subject_result_goal_ids),
            object_result_goal_ids=list(draft.object_result_goal_ids),
            goal=f"{draft.goal_id or 'entity'}:{draft.tool.value}",
            tool=draft.tool,
            subject_reference_indexes=subject_indexes,
            object_reference_indexes=object_indexes,
            relation_types=relation_types,
            raw_relation_types=raw_relation_types,
            direction=direction,
            target_types=target_types,
            requested_attributes=requested_attributes,
            depends_on=list(draft.depends_on),
        )

    def _validate_typed_references(self, decision: PlannerDecision) -> None:
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
                raise PlannerContractViolation("canonical_name_unknown")
            if matching_types and not matching_types.intersection(
                reference.expected_types
            ):
                raise PlannerContractViolation("canonical_name_wrong_type")

    def _validate_research_tasks(self, decision: PlannerDecision) -> None:
        allowed_raw = set(self.raw_relation_vocabulary)
        for task in decision.research_tasks:
            if allowed_raw and set(task.raw_relation_types) - allowed_raw:
                raise PlannerContractViolation("raw_relation_unknown")
            for index in [
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            ]:
                reference = decision.entity_references[index]
                if (
                    task.tool is ToolName.PERSONS
                    and NodeType.PERSON not in reference.expected_types
                ):
                    raise PlannerContractViolation("entity_task_wrong_type")
                if (
                    task.tool is ToolName.COMPANIES
                    and NodeType.COMPANY not in reference.expected_types
                ):
                    raise PlannerContractViolation("entity_task_wrong_type")

    @staticmethod
    def _validate_context_ids(decision: PlannerDecision, state: AgentState) -> None:
        allowed_ids = set(
            state.get("prior_focus_entity_ids", state.get("focus_entity_ids", []))
        )
        context_ids = {
            reference.context_entity_id
            for reference in decision.entity_references
            if reference.context_entity_id is not None
        }
        if context_ids - allowed_ids:
            raise PlannerContractViolation("context_id_untrusted")

    def _analysis_payload(
        self, state: AgentState, *, replan_count: int
    ) -> dict[str, Any]:
        terminal_candidate = state.get("planner_terminal_candidate")
        terminal_review = None
        if state.get("planner_terminal_review_pending") and terminal_candidate:
            candidate = PlannerAnalysisDecision.model_validate(terminal_candidate)
            terminal_review = {
                "required": True,
                "candidate": candidate.model_dump(mode="json"),
            }
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "recent_visible_turns": self._safe_recent_turns(state),
            "structured_summary": self._safe_summary(state),
            "prior_focus_context_set": self._safe_focus_context_set(state),
            "entity_catalog": list(self.entity_catalog),
            "raw_relation_vocabulary": list(self.raw_relation_vocabulary),
            "available_tools": list(self.available_tools),
            "contract_feedback": state.get("planner_contract_feedback"),
            "terminal_semantic_review": terminal_review,
            "current_replan_reason": state.get("current_replan_reason"),
            "replan_reason_history": list(state.get("replan_reasons", [])),
            "is_replan": bool(state.get("needs_replan")),
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

    def _tasks_payload(
        self,
        state: AgentState,
        analysis: PlannerAnalysisDecision,
        profiles: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "current_query": state.get("current_query", ""),
            "locale": state.get("locale", "zh-CN"),
            "validated_analysis": analysis.model_dump(mode="json"),
            "prior_focus_context_set": self._safe_focus_context_set(state),
            "entity_catalog": list(self.entity_catalog),
            "raw_relation_vocabulary": list(self.raw_relation_vocabulary),
            "available_tools": list(self.available_tools),
            "selected_example_profiles": list(profiles),
            "contract_feedback": state.get("planner_contract_feedback"),
            "current_replan_reason": state.get("current_replan_reason"),
            "limits": {
                "max_research_steps": self.max_research_steps,
                "remaining_research_steps": max(
                    0,
                    self.max_research_steps
                    - state.get("research_step_count", 0),
                ),
            },
        }

    @staticmethod
    def _safe_recent_turns(state: AgentState) -> list[dict[str, Any]]:
        safe: list[dict[str, Any]] = []
        for value in state.get("recent_turns", []):
            if isinstance(value, dict):
                user = value.get("user", "")
                assistant = value.get("assistant", "")
                intent = value.get("intent")
            else:
                user = getattr(value, "user", "")
                assistant = getattr(value, "assistant", "")
                intent = getattr(value, "intent", None)
            safe.append(
                {
                    "user": str(user),
                    "assistant": str(assistant),
                    "intent": getattr(intent, "value", intent),
                }
            )
        return safe

    @staticmethod
    def _safe_summary(state: AgentState) -> dict[str, Any]:
        value = state.get("summary")
        if isinstance(value, dict):
            getter = value.get
        else:
            def getter(key: str, default: Any = None) -> Any:
                return getattr(value, key, default)

        return {
            "user_goals": list(getter("user_goals", []) or []),
            "constraints": list(getter("constraints", []) or []),
            "unfinished_questions": list(
                getter("unfinished_questions", []) or []
            ),
            "summarized_turns": int(getter("summarized_turns", 0) or 0),
        }

    def _safe_focus_context_set(self, state: AgentState) -> dict[str, Any] | None:
        members = self._trusted_focus_members(state)
        if not members:
            return None
        return {
            "key": PlannerContextSetKey.PRIOR_FOCUS.value,
            "members": [
                {"name": member["name"], "entity_type": member["entity_type"]}
                for member in members
            ],
            "count": len(members),
        }

    @staticmethod
    def _trusted_focus_members(state: AgentState) -> list[dict[str, str]]:
        focus_ids = list(
            state.get("prior_focus_entity_ids", state.get("focus_entity_ids", []))
        )
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

    @staticmethod
    def _terminal_decision(analysis: PlannerAnalysisDecision) -> PlannerDecision:
        return PlannerDecision(
            intent=analysis.intent,
            entity_references=[],
            research_tasks=[],
            result_merge=ResultMergeStrategy.NOT_APPLICABLE,
            clarification_question=analysis.clarification_question,
            query_requires_realtime_data=analysis.query_requires_realtime_data,
        )

    @staticmethod
    def _model_call_updates(state: AgentState) -> dict[str, int]:
        return {
            "model_call_count": state.get("model_call_count", 0) + 1,
            "planner_model_calls": state.get("planner_model_calls", 0) + 1,
        }

    def _retry_or_fail(
        self,
        state: AgentState,
        *,
        route: list[str],
        stage: str,
        code: str,
        retry_field: str,
        retry_count: int,
        replan_count: int,
        call_updates: dict[str, int],
        contract_issues: tuple[ModelContractIssue, ...] = (),
    ) -> dict[str, Any]:
        attempts = retry_count + 1
        will_retry = attempts < 2
        logger.warning(
            "planner_stage_rejected",
            extra={
                "event": "planner_stage_rejected",
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id"),
                "stage": stage,
                "error_code": code,
                "will_retry": will_retry,
            },
        )
        feedback = self._contract_feedback(stage, code, contract_issues)
        update: dict[str, Any] = {
            **call_updates,
            "planner_decision": None,
            "planner_failed": True,
            retry_field: attempts,
            "planner_contract_retry_count": state.get(
                "planner_contract_retry_count", 0
            )
            + 1,
            "planner_contract_feedback": feedback,
            "run_status": "running" if will_retry else "failed",
            "needs_replan": False,
            "replan_count": replan_count,
            "route_history": route,
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step(f"{stage}_rejected", code),
            ],
        }
        if stage == "analysis":
            update.update(
                {
                    "planner_analysis": None,
                    "planner_task_plan": None,
                    "planner_prompt_profile": [],
                }
            )
        else:
            update.update({"planner_task_plan": None, "planner_prompt_profile": []})
        if not will_retry:
            update["llm_errors"] = [
                *state.get("llm_errors", []),
                f"Planner {stage} failed with {code}",
            ]
        return update

    @staticmethod
    def _contract_feedback(
        stage: str,
        code: str,
        issues: tuple[ModelContractIssue, ...],
    ) -> dict[str, str]:
        """Build one bounded retry hint from safe Schema metadata only."""

        feedback = {"stage": stage, "code": code}
        if not issues:
            field = _PLANNER_RUNTIME_FIELD_HINTS.get(code)
            if field is not None:
                feedback["field"] = field
                feedback["constraint"] = code
            return feedback
        issue = issues[0]
        feedback["field"] = _PLANNER_CONTRACT_FIELD_HINTS.get(
            issue.constraint,
            issue.field,
        )
        feedback["constraint"] = issue.constraint
        return feedback

    def _terminal_failure(
        self,
        state: AgentState,
        *,
        route: list[str],
        stage: str,
        code: str,
        retry_field: str,
        retry_count: int,
        replan_count: int,
    ) -> dict[str, Any]:
        return {
            "planner_analysis": None if stage == "analysis" else state.get("planner_analysis"),
            "planner_task_plan": None,
            "planner_prompt_profile": [],
            "planner_contract_feedback": {"stage": stage, "code": code},
            "planner_decision": None,
            "planner_failed": True,
            retry_field: retry_count,
            "run_status": "failed",
            "needs_replan": False,
            "replan_count": replan_count,
            "llm_errors": [*state.get("llm_errors", []), f"Planner {stage} failed with {code}"],
            "route_history": route,
            "agent_steps": [
                *state.get("agent_steps", []),
                self._safe_step(f"{stage}_rejected", code),
            ],
        }

    @staticmethod
    def _safe_step(
        action: str,
        error_code: str | None,
        decision: PlannerDecision | None = None,
    ) -> dict[str, Any]:
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
                decision.result_merge.value if decision is not None else None
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
            "count": len(decision.entity_references) if decision is not None else 0,
            "error_code": error_code,
        }
