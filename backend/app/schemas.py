"""Canonical contracts shared by agents, tools, memory, API, and frontend."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class NodeType(StrEnum):
    PERSON = "person"
    COMPANY = "company"
    LOCATION = "location"


class RelationType(StrEnum):
    CONTROLS = "controls"
    FOUNDED = "founded"
    WORKS_AT = "works_at"
    RELATED_TO = "related_to"
    HEADQUARTERED_IN = "headquartered_in"
    PARTNER_OF = "partner_of"
    SUPPLIER_TO = "supplier_to"
    INVESTED_IN = "invested_in"
    OWNS = "owns"


class Intent(StrEnum):
    FIND_CONTROLLED_COMPANIES = "find_controlled_companies"
    FIND_RELATED_COMPANIES = "find_related_companies"
    LOCATE_ENTITIES = "locate_entities"
    GET_COMPANY_PROFILE = "get_company_profile"
    GET_PERSON_PROFILE = "get_person_profile"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class ToolName(StrEnum):
    COMPANIES = "companies"
    PERSONS = "persons"
    RELATIONS = "relations"


class CacheScope(StrEnum):
    """Whether a cached result is safe outside the conversation that produced it."""

    CONTEXT_FREE = "context_free"
    CONVERSATION = "conversation"


class EntityReferenceSource(StrEnum):
    """Where Planner obtained the identity behind an entity mention."""

    CURRENT_QUERY = "current_query"
    CONVERSATION_CONTEXT = "conversation_context"


class EntityReferenceRole(StrEnum):
    """Semantic endpoint role assigned by Planner, never a factual assertion."""

    SUBJECT = "subject"
    OBJECT = "object"


class ResultMergeStrategy(StrEnum):
    """How Planner wants independently researched task results combined."""

    NOT_APPLICABLE = "not_applicable"
    UNION = "union"
    INTERSECTION = "intersection"
    DIRECT = "direct"


class ResearchDirection(StrEnum):
    """Direction requested by a Planner research task."""

    NOT_APPLICABLE = "not_applicable"
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    ANY = "any"


class ControlQueryPolicy(StrEnum):
    """How an explicit natural-language control request should be researched."""

    NOT_APPLICABLE = "not_applicable"
    EXPLICIT_ONLY = "explicit_only"
    EXPLICIT_THEN_STRONG_ASSOCIATIONS = "explicit_then_strong_associations"


class ResearchAction(StrEnum):
    """The next action chosen by the prompt-driven Researcher."""

    CALL_TOOL = "call_tool"
    FINISH = "finish"
    NO_RESULTS = "no_results"
    REPLAN = "replan"
    FAIL = "fail"


class CacheStatus(StrEnum):
    WARM = "warm"
    HOT = "hot"
    STALE = "stale"


class MemoryOperation(StrEnum):
    ADD = "add"
    TOUCH = "touch"
    PROMOTE = "promote"
    SKIP = "skip"
    NONE = "none"


class ChatStatus(StrEnum):
    SUCCESS = "success"
    CLARIFICATION = "clarification"
    FAILED = "failed"


class ChatErrorCode(StrEnum):
    MODEL_FAILURE = "model_failure"
    PLANNING_FAILURE = "planning_failure"
    RESEARCH_FAILURE = "research_failure"
    TOOL_FAILURE = "tool_failure"
    AGENT_FAILURE = "agent_failure"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    provider: str
    record_id: str
    source_kind: str
    updated_at: datetime
    retrieved_at: datetime = Field(default_factory=utc_now)
    is_demo: bool = True
    source_url: str | None = None


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: NodeType
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    type: RelationType
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class GraphPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    data_version: str

    @field_validator("evidence", mode="after")
    @classmethod
    def evidence_is_a_unique_catalog(cls, value: list[Evidence]) -> list[Evidence]:
        by_id: dict[str, Evidence] = {}
        for item in value:
            previous = by_id.get(item.id)
            if previous is not None:
                previous_stable = previous.model_dump(exclude={"retrieved_at"})
                item_stable = item.model_dump(exclude={"retrieved_at"})
                if previous_stable != item_stable:
                    raise ValueError(
                        f"conflicting evidence records share id: {item.id}"
                    )
                if previous.retrieved_at >= item.retrieved_at:
                    continue
            by_id[item.id] = item
        return sorted(by_id.values(), key=lambda item: item.id)

    @model_validator(mode="after")
    def graph_references_exist(self) -> GraphPayload:
        node_ids = {node.id for node in self.nodes}
        missing = {
            endpoint
            for edge in self.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in node_ids
        }
        if missing:
            raise ValueError(f"graph edges reference missing nodes: {sorted(missing)}")

        referenced_evidence_ids = {
            evidence_id
            for element in [*self.nodes, *self.edges]
            for evidence_id in element.evidence_ids
        }
        catalog_ids = {item.id for item in self.evidence}
        missing_evidence = referenced_evidence_ids - catalog_ids
        if missing_evidence:
            raise ValueError(
                "graph elements reference missing evidence: "
                f"{sorted(missing_evidence)}"
            )
        unused_evidence = catalog_ids - referenced_evidence_ids
        if unused_evidence:
            raise ValueError(
                "graph evidence catalog contains unreferenced records: "
                f"{sorted(unused_evidence)}"
            )
        return self


class EntityReference(BaseModel):
    """Planner's typed interpretation of one literal mention in the current query.

    A context-backed reference may reuse exactly one already verified ID. A newly
    named entity is intentionally ID-free until Researcher resolves it with a mock
    entity tool.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mention: str = Field(min_length=1, max_length=200)
    source: EntityReferenceSource
    role: EntityReferenceRole
    expected_types: list[NodeType] = Field(min_length=1, max_length=3)
    canonical_name: str | None = Field(default=None, min_length=1, max_length=200)
    context_entity_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("expected_types", mode="after")
    @classmethod
    def unique_expected_types(cls, value: list[NodeType]) -> list[NodeType]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def identity_source_matches_id(self) -> EntityReference:
        if self.source is EntityReferenceSource.CURRENT_QUERY:
            if self.context_entity_id is not None:
                raise ValueError("a newly named entity cannot carry a context ID")
        elif self.context_entity_id is None:
            raise ValueError("a conversation-context reference requires a verified ID")
        return self


class ResearchTask(BaseModel):
    """One Planner-authored unit in the Researcher task DAG.

    References use indexes into ``PlannerDecision.entity_references`` so a task can
    depend on an entity that has not yet been resolved to a stable tool ID.  Planner
    chooses goals and tool capabilities; Researcher remains responsible for calling
    the tool and obtaining the verified ID/records.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    goal: str = Field(min_length=1, max_length=500)
    tool: ToolName
    subject_reference_indexes: list[int] = Field(max_length=100)
    object_reference_indexes: list[int] = Field(max_length=100)
    relation_types: list[RelationType] = Field(max_length=20)
    raw_relation_types: list[str] = Field(max_length=50)
    direction: ResearchDirection
    target_types: list[NodeType] = Field(max_length=3)
    requested_attributes: list[str] = Field(max_length=30)
    depends_on: list[str] = Field(max_length=100)

    @field_validator(
        "subject_reference_indexes",
        "object_reference_indexes",
        mode="after",
    )
    @classmethod
    def unique_non_negative_indexes(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError("reference indexes must be non-negative")
        return list(dict.fromkeys(value))

    @field_validator(
        "relation_types",
        "raw_relation_types",
        "target_types",
        "requested_attributes",
        "depends_on",
        mode="after",
    )
    @classmethod
    def unique_task_lists(cls, value: list[Any]) -> list[Any]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def coherent_task_shape(self) -> ResearchTask:
        if self.task_id in self.depends_on:
            raise ValueError("a research task cannot depend on itself")
        if self.tool is not ToolName.RELATIONS and (
            self.relation_types
            or self.raw_relation_types
            or self.direction is not ResearchDirection.NOT_APPLICABLE
        ):
            raise ValueError("only a relations task may define a relation scope")
        if (
            self.tool is ToolName.RELATIONS
            and self.direction is ResearchDirection.NOT_APPLICABLE
        ):
            raise ValueError("a relations task requires a direction")
        return self


class QuerySignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 4
    intent: Intent = Field(
        description=(
            "用户的整体信息目标；它决定任务目标，但不得替代 research_tasks 的具体拆解。"
        )
    )
    subject_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    requested_relation_types: list[RelationType] = Field(default_factory=list)
    effective_relation_types: list[RelationType] = Field(default_factory=list)
    raw_relation_qualifiers: list[str] = Field(default_factory=list)
    verified_empty_relation_types: list[RelationType] = Field(default_factory=list)
    target_types: list[NodeType] = Field(default_factory=list)
    requested_attributes: list[str] = Field(default_factory=list)
    context_entity_ids: list[str] = Field(default_factory=list)
    result_merge: ResultMergeStrategy = ResultMergeStrategy.NOT_APPLICABLE
    control_policy: ControlQueryPolicy = ControlQueryPolicy.NOT_APPLICABLE
    control_policy_version: str = Field(
        default="control-policy-v1", min_length=1, max_length=80
    )
    entity_match_version: str = Field(
        default="entity-match-v1", min_length=1, max_length=80
    )
    locale: str = "zh-CN"

    @field_validator(
        "subject_ids",
        "object_ids",
        "relation_types",
        "requested_relation_types",
        "effective_relation_types",
        "raw_relation_qualifiers",
        "verified_empty_relation_types",
        "target_types",
        "requested_attributes",
        "context_entity_ids",
        mode="after",
    )
    @classmethod
    def sorted_unique(cls, value: list[Any]) -> list[Any]:
        return sorted(set(value), key=str)

    @model_validator(mode="after")
    def coherent_relation_semantics(self) -> QuerySignature:
        if set(self.relation_types) != set(self.effective_relation_types):
            raise ValueError(
                "relation_types and effective_relation_types must remain equivalent"
            )
        known_scope = {
            *self.requested_relation_types,
            *self.effective_relation_types,
        }
        if set(self.verified_empty_relation_types) - known_scope:
            raise ValueError(
                "verified-empty relations must belong to requested or effective scope"
            )
        if self.intent is Intent.FIND_CONTROLLED_COMPANIES:
            if self.control_policy is ControlQueryPolicy.NOT_APPLICABLE:
                raise ValueError("control queries require an explicit control policy")
        elif self.control_policy is not ControlQueryPolicy.NOT_APPLICABLE:
            raise ValueError("control policy is only valid for control queries")
        return self


class PlannerDecision(BaseModel):
    """Planner-authored entity alignment and executable research task DAG."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Intent = Field(
        description=(
            "整体查询意图。find_controlled_companies 只用于用户明确询问‘控制/control’；"
            "‘拥有/持有/own’属于 find_related_companies，并由 owns 关系任务表达。"
        )
    )
    entity_references: list[EntityReference] = Field(max_length=100)
    research_tasks: list[ResearchTask] = Field(max_length=100)
    result_merge: ResultMergeStrategy
    clarification_question: str | None = Field(max_length=500)
    query_requires_realtime_data: bool

    @model_validator(mode="after")
    def coherent_planner_decision(self) -> PlannerDecision:
        terminal = self.intent in {Intent.CLARIFY, Intent.UNSUPPORTED} or (
            self.query_requires_realtime_data
        )
        needs_clarification = self.intent is Intent.CLARIFY
        if needs_clarification and not self.clarification_question:
            raise ValueError("clarification_question is required for clarification")
        if not needs_clarification and self.clarification_question:
            raise ValueError("only clarification may include a clarification question")
        if terminal and self.research_tasks:
            raise ValueError("terminal Planner decisions cannot include research tasks")
        if not terminal and not self.research_tasks:
            raise ValueError("an executable Planner decision requires research tasks")

        task_ids = [task.task_id for task in self.research_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("research task IDs must be unique")
        known_task_ids = set(task_ids)
        reference_count = len(self.entity_references)
        dependency_graph: dict[str, set[str]] = {}
        for task in self.research_tasks:
            indexes = {
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            }
            if any(index >= reference_count for index in indexes):
                raise ValueError("research task references an unknown entity index")
            if set(task.depends_on) - known_task_ids:
                raise ValueError("research task depends on an unknown task")
            dependency_graph[task.task_id] = set(task.depends_on)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("research task dependencies must be acyclic")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependency_graph.get(task_id, set()):
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)
        return self


class ResearcherDecision(BaseModel):
    """Minimal model-selected Researcher transition.

    The model chooses one fact-tool call or one lifecycle transition.  Native
    finish/no-results calls are argument-free signals; runtime projects their
    supporting record IDs from the successful typed tool receipt.  Scripted tests
    may still provide explicit IDs to exercise negative validation paths.  A
    canonical query signature, entity bindings, and follow-up focus are deterministic
    projections of Planner semantics plus successful tool receipts and therefore do
    not belong in the provider output schema.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: ResearchAction = Field(
        description=(
            "当范围正确的 relations 调用成功返回零条记录且查询实体已经验证时，"
            "选择 no_results，而不是 fail 或 finish。"
        )
    )
    tool: ToolName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    selected_record_ids: list[str] = Field(default_factory=list, max_length=300)
    failure_message: str | None = Field(default=None, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_action_branch(cls, value: Any) -> Any:
        """Keep only fields that are meaningful for the selected action branch.

        Function-calling models sometimes populate fields from more than one optional
        schema branch while choosing the correct action. Those fields are not facts
        and runtime never uses them. Canonicalizing the inactive branches preserves
        the one-action contract without weakening active tool arguments, required
        finish signatures, or replan/fail reasons.
        """

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        action = normalized.get("action")
        action_value = action.value if isinstance(action, ResearchAction) else action
        if action_value == ResearchAction.CALL_TOOL.value:
            return {
                **normalized,
                "selected_record_ids": [],
                "failure_message": None,
            }
        if action_value in {
            ResearchAction.FINISH.value,
            ResearchAction.NO_RESULTS.value,
        }:
            return {
                **normalized,
                "tool": None,
                "arguments": {},
                "failure_message": None,
            }
        if action_value in {
            ResearchAction.REPLAN.value,
            ResearchAction.FAIL.value,
        }:
            return {
                **normalized,
                "tool": None,
                "arguments": {},
                "selected_record_ids": [],
            }
        return normalized

    @field_validator("selected_record_ids", mode="after")
    @classmethod
    def unique_research_lists(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def coherent_research_action(self) -> ResearcherDecision:
        if self.action is ResearchAction.CALL_TOOL:
            if self.tool is None:
                raise ValueError("tool is required for call_tool")
        else:
            if self.tool is not None or self.arguments:
                raise ValueError("only call_tool may provide a tool or arguments")
        if self.action in {ResearchAction.REPLAN, ResearchAction.FAIL} and not self.failure_message:
            raise ValueError("replan/fail requires failure_message")
        return self

class VisualizerDecision(BaseModel):
    """Minimal Visualizer output: localized prose plus its verified support IDs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=4000)
    answer_record_ids: list[str] = Field(default_factory=list, max_length=300)

    @field_validator("answer_record_ids", mode="after")
    @classmethod
    def unique_visualizer_lists(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class VisualizerTextOnlyDecision(VisualizerDecision):
    """Phase-scoped Visualizer output for clarification and verified empty results.

    The model still authors the localized answer, while the provider schema prevents
    it from confusing graph-retained subject nodes with records that evidence an
    answer.  The runtime applies the same base ``VisualizerDecision`` validation and
    verified-empty/clarification invariants after parsing.
    """

    answer_record_ids: list[str] = Field(default_factory=list, max_length=0)


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    tool: ToolName
    provider: str
    data_version: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    elapsed_ms: int = Field(ge=0)
    error: ToolError | None = None


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str
    assistant: str
    created_at: datetime = Field(default_factory=utc_now)
    intent: Intent | None = None
    focus_entity_ids: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_goals: list[str] = Field(default_factory=list)
    resolved_entities: dict[str, str] = Field(default_factory=dict)
    focus_entity_ids: list[str] = Field(default_factory=list)
    confirmed_fact_ids: list[str] = Field(default_factory=list)
    confirmed_evidence_ids: list[str] = Field(default_factory=list)
    latest_graph_id: str | None = None
    constraints: list[str] = Field(default_factory=list)
    unfinished_questions: list[str] = Field(default_factory=list)
    summarized_turns: int = 0
    updated_at: datetime = Field(default_factory=utc_now)


class CacheMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_hit: bool = False
    tier: str | None = None
    match_type: str | None = None
    status: CacheStatus | None = None
    write_operation: MemoryOperation = MemoryOperation.NONE
    result_id: str | None = None
    reason: str | None = None


class AgentTraceRole(StrEnum):
    """Public Agent roles allowed in the bounded execution trace."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    VISUALIZER = "visualizer"


class EntityResolutionTraceStrategy(StrEnum):
    """Auditable entity-search strategies safe to expose in the public trace."""

    EXACT = "exact"
    FUZZY = "fuzzy"


class AgentStepTrace(BaseModel):
    """One safe, typed execution step suitable for the public API.

    The trace deliberately excludes queries, prompts, model payloads, tool results,
    free-form reasons, and entity properties.  It carries only bounded control-flow
    metadata and stable identifiers that have already crossed the runtime's factual
    validation boundary.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: AgentTraceRole
    action: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    tool: ToolName | None = None
    relation_types: list[RelationType] = Field(default_factory=list, max_length=20)
    result_merge: ResultMergeStrategy | None = None
    resolution_strategy: EntityResolutionTraceStrategy | None = None
    resolution_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$",
    )
    record_ids: list[str] = Field(default_factory=list, max_length=128)
    argument_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    count: int = Field(default=0, ge=0, le=1_000_000)
    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )

    @field_validator("relation_types", mode="after")
    @classmethod
    def unique_relation_types(
        cls, value: list[RelationType]
    ) -> list[RelationType]:
        if len(value) != len(set(value)):
            raise ValueError("agent trace relation_types must be unique")
        return value

    @model_validator(mode="after")
    def coherent_entity_resolution_trace(self) -> AgentStepTrace:
        if self.result_merge is not None and (
            self.role is not AgentTraceRole.PLANNER or self.action != "plan"
        ):
            raise ValueError(
                "result_merge is valid only on an accepted Planner plan"
            )
        has_strategy = self.resolution_strategy is not None
        has_version = self.resolution_version is not None
        if has_strategy != has_version:
            raise ValueError(
                "entity resolution strategy and version must be reported together"
            )
        if has_strategy and (
            self.role is not AgentTraceRole.RESEARCHER
            or self.action != "tool_result"
            or self.tool not in {ToolName.PERSONS, ToolName.COMPANIES}
        ):
            raise ValueError(
                "entity resolution metadata is valid only on entity tool results"
            )
        return self

    @field_validator("record_ids", mode="after")
    @classmethod
    def safe_stable_record_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("agent trace record_ids must be unique")
        allowed_namespaces = {"person", "company", "location", "relation"}
        for record_id in value:
            if not (3 <= len(record_id) <= 220):
                raise ValueError("agent trace record ID has an invalid length")
            namespace, separator, suffix = record_id.partition(":")
            if namespace not in allowed_namespaces or not separator or not suffix:
                raise ValueError("agent trace record ID has an invalid namespace")
            if any(
                not (character.isalnum() or character in {".", "_", ":", "-"})
                for character in suffix
            ):
                raise ValueError("agent trace record ID contains unsafe characters")
        return value


class TraceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    researcher_invoked: bool = False
    tool_calls: int = 0
    research_steps: int = 0
    replans: int = 0
    model_provider: str = "openai"
    model_name: str | None = None
    model_calls: int = 0
    planner_model_calls: int = 0
    researcher_model_calls: int = 0
    visualizer_model_calls: int = 0
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    route_history: list[str] = Field(default_factory=list)
    agent_steps: list[AgentStepTrace] = Field(default_factory=list, max_length=64)


class CachedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    graph: GraphPayload
    evidence: list[Evidence]
    query_signature: QuerySignature
    focus_entity_ids: list[str] = Field(default_factory=list)
    resolved_entities: dict[str, str] = Field(default_factory=dict)
    cache_scope: CacheScope = CacheScope.CONTEXT_FREE


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=1000)
    locale: str = Field(default="zh-CN", min_length=2, max_length=16)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    request_id: str
    status: ChatStatus
    error_code: ChatErrorCode | None = None
    answer: str
    graph_id: str
    graph: GraphPayload
    memory: CacheMetadata
    trace: TraceMetadata
    disclaimer: str = "结果来自本地演示数据，不代表实时工商或法律结论。"

    @model_validator(mode="after")
    def status_and_error_are_coherent(self) -> ChatResponse:
        if self.status is ChatStatus.FAILED and self.error_code is None:
            raise ValueError("failed responses require a safe error_code")
        if self.status is not ChatStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed responses may carry error_code")
        return self


class HealthResponse(BaseModel):
    status: str
    service: str = "enterprise-relationship-explorer"


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]
