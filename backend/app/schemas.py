"""Canonical contracts shared by agents, tools, memory, API, and frontend."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError


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
    MULTI_GOAL = "multi_goal"
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


class RequestedAttribute(StrEnum):
    """Closed vocabulary shared by Planner profile goals and fact tools.

    A single canonical vocabulary prevents the provider-facing Planner Schema and
    the executable entity-tool Schemas from drifting apart.  Individual tools
    expose type-specific subsets derived from this enum below.
    """

    SOURCE_ID = "source_id"
    ALIASES = "aliases"
    NATIONALITY = "nationality"
    SUMMARY = "summary"
    FOUNDED_YEAR = "founded_year"
    LEGAL_REP_ID = "legal_rep_id"
    CITY = "city"
    LOCATION_ID = "location_id"
    LOCATION = "location"
    DEMO_DATA = "demo_data"


PERSON_REQUESTED_ATTRIBUTES = frozenset(
    {
        RequestedAttribute.SOURCE_ID,
        RequestedAttribute.ALIASES,
        RequestedAttribute.NATIONALITY,
        RequestedAttribute.SUMMARY,
        RequestedAttribute.DEMO_DATA,
    }
)
COMPANY_REQUESTED_ATTRIBUTES = frozenset(
    {
        RequestedAttribute.SOURCE_ID,
        RequestedAttribute.ALIASES,
        RequestedAttribute.FOUNDED_YEAR,
        RequestedAttribute.LEGAL_REP_ID,
        RequestedAttribute.CITY,
        RequestedAttribute.LOCATION_ID,
        RequestedAttribute.LOCATION,
        RequestedAttribute.DEMO_DATA,
    }
)


def _attribute_subset_enum(
    name: str, values: frozenset[RequestedAttribute]
) -> type[StrEnum]:
    """Build a tool-specific enum from the canonical Planner vocabulary."""

    return StrEnum(
        name,
        {
            attribute.name: attribute.value
            for attribute in RequestedAttribute
            if attribute in values
        },
        module=__name__,
    )


# Kept as distinct enum types so each ToolSpec advertises only attributes that
# its handler can execute, while both are mechanically derived from the same
# provider-facing vocabulary.
PersonAttribute = _attribute_subset_enum(
    "PersonAttribute", PERSON_REQUESTED_ATTRIBUTES
)
CompanyAttribute = _attribute_subset_enum(
    "CompanyAttribute", COMPANY_REQUESTED_ATTRIBUTES
)


class CacheScope(StrEnum):
    """Whether a cached result is safe outside the conversation that produced it."""

    CONTEXT_FREE = "context_free"
    CONVERSATION = "conversation"


class EntityReferenceSource(StrEnum):
    """Where Planner obtained the identity behind an entity mention."""

    CURRENT_QUERY = "current_query"
    CONVERSATION_CONTEXT = "conversation_context"


class PlannerContextSetKey(StrEnum):
    """Trusted conversation sets a Planner may reference without copying IDs."""

    PRIOR_FOCUS = "prior_focus"


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


class GoalResultGrouping(StrEnum):
    """Whether an equivalent-scope goal belongs to a merged or separate result."""

    MERGED = "merged"
    SEPARATE = "separate"


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


class TaskScopeSource(StrEnum):
    """Typed source from which runtime projects a task's factual scope."""

    NOT_APPLICABLE = "not_applicable"
    GOAL = "goal"
    CONTROL_EXPLICIT = "control_explicit"
    CONTROL_FALLBACK = "control_fallback"


class GoalResultStatus(StrEnum):
    """Evidence-backed completion state for one canonical research goal."""

    NONEMPTY = "nonempty"
    VERIFIED_EMPTY = "verified_empty"
    SKIPPED_EMPTY_INPUT = "skipped_empty_input"


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


class PlannerAnalysisEntityReference(BaseModel):
    """Provider-side entity reference produced by Planner's analysis stage.

    Conversation references name a trusted set.  Stable IDs never cross this
    model boundary; runtime expands the set only after the task plan validates.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mention: str = Field(
        min_length=1,
        max_length=200,
        description="用户当前问题中的原文提及；代词必须保留原文。",
    )
    source: EntityReferenceSource = Field(
        description=(
            "明确的新名称使用 current_query；指向已验证上轮焦点的代词使用 "
            "conversation_context。"
        )
    )
    role: EntityReferenceRole
    expected_types: list[NodeType] = Field(min_length=1, max_length=3)
    canonical_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "current_query 可选择实体目录中的标准名；会话集合引用必须为 null。"
        ),
    )
    context_set_key: PlannerContextSetKey | None = Field(
        default=None,
        description=(
            "仅 conversation_context 可使用，且当前唯一允许值为 prior_focus；"
            "current_query 必须为 null。"
        ),
    )

    @field_validator("expected_types", mode="after")
    @classmethod
    def unique_analysis_expected_types(
        cls, value: list[NodeType]
    ) -> list[NodeType]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def context_source_matches_set(self) -> PlannerAnalysisEntityReference:
        if self.source is EntityReferenceSource.CURRENT_QUERY:
            if self.context_set_key is not None:
                raise ValueError("a current-query reference cannot select a context set")
        else:
            if self.context_set_key is not PlannerContextSetKey.PRIOR_FOCUS:
                raise ValueError(
                    "a conversation-context reference must select prior_focus"
                )
            if self.canonical_name is not None:
                raise ValueError(
                    "a conversation-context set cannot carry one canonical name"
                )
        return self


class ResearchGoal(BaseModel):
    """One semantic research objective over any number of entity references."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="当前分析内唯一的语义目标标识。",
    )
    intent: Intent = Field(
        description="该目标的事实意图；不能使用 multi_goal、clarify 或 unsupported。"
    )
    subject_reference_indexes: list[int] = Field(
        default_factory=list,
        max_length=100,
        description="同一目标的全部直接主体索引；不得按实体拆成重复目标。",
    )
    object_reference_indexes: list[int] = Field(
        default_factory=list,
        max_length=100,
        description="明确客体索引；direct 使用与主体相同的完整操作数集合。",
    )
    subject_result_goal_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="仅当用户要求继续研究前序结果时填写的主体结果集。",
    )
    object_result_goal_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="仅当用户要求以前序结果作为客体时填写。",
    )
    relation_types: list[RelationType] = Field(
        default_factory=list,
        max_length=20,
        description="用户明确限定的 typed 关系；空列表表示完整业务关系范围。",
    )
    raw_relation_types: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="用户明确限定的原始关系词；否则保持为空。",
    )
    direction: ResearchDirection = Field(
        default=ResearchDirection.NOT_APPLICABLE,
        description="关系相对 subject 的方向；关系目标不能为 not_applicable。",
    )
    target_types: list[NodeType] = Field(
        default_factory=list,
        max_length=3,
        description="结果端点类型；direct 必须为空。",
    )
    requested_attributes: list[RequestedAttribute] = Field(
        default_factory=list,
        max_length=30,
        description=(
            "资料或属性目标明确要求的闭合属性键；人物和企业资料分别受对应工具能力"
            "子集约束。"
        ),
    )
    aggregation: ResultMergeStrategy = Field(
        default=ResultMergeStrategy.NOT_APPLICABLE,
        description=(
            "同一目标内 N 个操作数的集合语义：union 返回连接至少一个操作数的外部"
            "邻居；intersection 返回连接每个操作数的共同外部邻居；direct 只返回"
            "操作数集合内部端点之间的边，不扩展外部邻居。"
        ),
    )
    result_grouping: GoalResultGrouping = Field(
        default=GoalResultGrouping.MERGED,
        description=(
            "同范围操作数通常使用 merged 并进入一个 N 元目标；只有用户明确要求"
            "保留互相独立、可分别回答的结果组时，该组对应的目标才使用 separate。"
        ),
    )
    control_policy: ControlQueryPolicy = Field(
        default=ControlQueryPolicy.NOT_APPLICABLE,
        description="只有控制目标使用受控两阶段策略。",
    )
    depends_on_goal_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="本目标真实消费或依赖的前序 goal_id；必须无环。",
    )

    @field_validator(
        "subject_reference_indexes",
        "object_reference_indexes",
        mode="after",
    )
    @classmethod
    def unique_goal_indexes(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("goal reference indexes must be non-negative")
        return list(dict.fromkeys(value))

    @field_validator(
        "relation_types",
        "raw_relation_types",
        "target_types",
        "requested_attributes",
        "depends_on_goal_ids",
        "subject_result_goal_ids",
        "object_result_goal_ids",
        mode="after",
    )
    @classmethod
    def unique_goal_lists(cls, value: list[Any]) -> list[Any]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def coherent_goal(self) -> ResearchGoal:
        if self.goal_id in self.depends_on_goal_ids:
            raise PydanticCustomError(
                "goal_self_dependency",
                "goal_self_dependency",
            )
        if not self.subject_reference_indexes and not self.subject_result_goal_ids:
            raise PydanticCustomError(
                "goal_subjects_required",
                "goal_subjects_required",
            )
        if (
            set(self.subject_result_goal_ids) | set(self.object_result_goal_ids)
        ) - set(self.depends_on_goal_ids):
            raise PydanticCustomError(
                "goal_result_dependency_required",
                "goal_result_dependency_required",
            )
        if self.intent in {Intent.CLARIFY, Intent.UNSUPPORTED, Intent.MULTI_GOAL}:
            raise PydanticCustomError(
                "goal_intent_not_executable",
                "goal_intent_not_executable",
            )
        if self.intent is Intent.FIND_CONTROLLED_COMPANIES:
            if (
                self.control_policy
                is not ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS
            ):
                raise PydanticCustomError(
                    "control_policy_required",
                    "control_policy_required",
                )
            if self.relation_types != [RelationType.CONTROLS]:
                raise PydanticCustomError(
                    "control_scope_must_start_explicit",
                    "control_scope_must_start_explicit",
                )
        elif self.control_policy is not ControlQueryPolicy.NOT_APPLICABLE:
            raise PydanticCustomError(
                "control_policy_not_applicable",
                "control_policy_not_applicable",
            )
        if self.intent is Intent.LOCATE_ENTITIES:
            if set(self.relation_types) != {RelationType.HEADQUARTERED_IN}:
                raise PydanticCustomError(
                    "location_relation_scope_required",
                    "location_relation_scope_required",
                )
            if self.direction is not ResearchDirection.OUTGOING:
                raise PydanticCustomError(
                    "location_direction_must_be_outgoing",
                    "location_direction_must_be_outgoing",
                )
            if NodeType.LOCATION not in self.target_types:
                raise PydanticCustomError(
                    "location_target_type_required",
                    "location_target_type_required",
                )
        profile = self.intent in {
            Intent.GET_PERSON_PROFILE,
            Intent.GET_COMPANY_PROFILE,
        }
        if profile:
            if (
                self.relation_types
                or self.raw_relation_types
                or self.direction is not ResearchDirection.NOT_APPLICABLE
            ):
                raise PydanticCustomError(
                    "profile_relation_scope_forbidden",
                    "profile_relation_scope_forbidden",
                )
            allowed_attributes = (
                PERSON_REQUESTED_ATTRIBUTES
                if self.intent is Intent.GET_PERSON_PROFILE
                else COMPANY_REQUESTED_ATTRIBUTES
            )
            if set(self.requested_attributes) - allowed_attributes:
                raise PydanticCustomError(
                    "profile_attribute_not_supported",
                    "profile_attribute_not_supported",
                )
        elif self.direction is ResearchDirection.NOT_APPLICABLE:
            raise PydanticCustomError(
                "relation_direction_required",
                "relation_direction_required",
            )
        operand_count = len(
            {*self.subject_reference_indexes, *self.object_reference_indexes}
        )
        nary_relation_intents = {
            Intent.FIND_RELATED_COMPANIES,
            Intent.FIND_CONTROLLED_COMPANIES,
        }
        if (
            self.intent in nary_relation_intents
            and operand_count > 1
            and self.aggregation is ResultMergeStrategy.NOT_APPLICABLE
        ):
            raise PydanticCustomError(
                "nary_aggregation_required",
                "nary_aggregation_required",
            )
        if (
            self.intent is Intent.LOCATE_ENTITIES
            and self.aggregation is not ResultMergeStrategy.NOT_APPLICABLE
        ):
            raise PydanticCustomError(
                "location_aggregation_not_applicable",
                "location_aggregation_not_applicable",
            )
        if (
            self.aggregation is ResultMergeStrategy.DIRECT
            and self.target_types
        ):
            raise PydanticCustomError(
                "direct_target_types_must_be_empty",
                "direct_target_types_must_be_empty",
            )
        return self


class PlannerTaskDraft(BaseModel):
    """Provider-side DAG node; factual scope is inherited from one typed goal."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    goal_id: str | None = Field(default=None, min_length=1, max_length=80)
    subject_result_goal_ids: list[str] = Field(default_factory=list, max_length=100)
    object_result_goal_ids: list[str] = Field(default_factory=list, max_length=100)
    tool: ToolName
    subject_reference_indexes: list[int] = Field(default_factory=list, max_length=100)
    object_reference_indexes: list[int] = Field(default_factory=list, max_length=100)
    scope_source: TaskScopeSource = TaskScopeSource.NOT_APPLICABLE
    depends_on: list[str] = Field(default_factory=list, max_length=100)

    @field_validator(
        "subject_reference_indexes",
        "object_reference_indexes",
        mode="after",
    )
    @classmethod
    def unique_draft_indexes(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("task reference indexes must be non-negative")
        return list(dict.fromkeys(value))

    @field_validator("depends_on", mode="after")
    @classmethod
    def unique_draft_dependencies(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator(
        "subject_result_goal_ids", "object_result_goal_ids", mode="after"
    )
    @classmethod
    def unique_draft_result_sets(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def coherent_draft(self) -> PlannerTaskDraft:
        if self.task_id in self.depends_on:
            raise ValueError("a task cannot depend on itself")
        if self.tool is ToolName.RELATIONS:
            if self.goal_id is None or self.scope_source is TaskScopeSource.NOT_APPLICABLE:
                raise ValueError("relations drafts require a goal and scope source")
        elif self.scope_source is not TaskScopeSource.NOT_APPLICABLE:
            raise ValueError("entity drafts cannot select a relation scope")
        if (
            set(self.subject_result_goal_ids) | set(self.object_result_goal_ids)
        ) and self.tool is not ToolName.RELATIONS:
            raise ValueError("only relations drafts may consume goal result sets")
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
    goal_id: str | None = Field(default=None, min_length=1, max_length=80)
    subject_result_goal_ids: list[str] = Field(default_factory=list, max_length=100)
    object_result_goal_ids: list[str] = Field(default_factory=list, max_length=100)
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
        "subject_result_goal_ids",
        "object_result_goal_ids",
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


class QueryGoalSignature(BaseModel):
    """Canonical, evidence-backed signature for one Planner research goal."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(min_length=1, max_length=80)
    intent: Intent
    subject_ids: list[str] = Field(default_factory=list, max_length=100)
    object_ids: list[str] = Field(default_factory=list, max_length=100)
    relation_types: list[RelationType] = Field(default_factory=list, max_length=20)
    requested_relation_types: list[RelationType] = Field(default_factory=list, max_length=20)
    effective_relation_types: list[RelationType] = Field(default_factory=list, max_length=20)
    raw_relation_qualifiers: list[str] = Field(default_factory=list, max_length=50)
    verified_empty_relation_types: list[RelationType] = Field(default_factory=list, max_length=20)
    target_types: list[NodeType] = Field(default_factory=list, max_length=3)
    requested_attributes: list[str] = Field(default_factory=list, max_length=30)
    aggregation: ResultMergeStrategy = ResultMergeStrategy.NOT_APPLICABLE
    control_policy: ControlQueryPolicy = ControlQueryPolicy.NOT_APPLICABLE
    depends_on_goal_ids: list[str] = Field(default_factory=list, max_length=100)
    context_entity_ids: list[str] = Field(default_factory=list, max_length=100)
    result_status: GoalResultStatus
    result_record_ids: list[str] = Field(default_factory=list, max_length=300)
    focus_entity_ids: list[str] = Field(default_factory=list, max_length=100)

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
        "depends_on_goal_ids",
        "context_entity_ids",
        "result_record_ids",
        "focus_entity_ids",
        mode="after",
    )
    @classmethod
    def canonical_goal_lists(cls, value: list[Any]) -> list[Any]:
        return sorted(set(value), key=str)

    @model_validator(mode="after")
    def coherent_goal_signature(self) -> QueryGoalSignature:
        if set(self.relation_types) != set(self.effective_relation_types):
            raise ValueError("goal relation types must equal effective relation types")
        known_scope = {*self.requested_relation_types, *self.effective_relation_types}
        if set(self.verified_empty_relation_types) - known_scope:
            raise ValueError("goal verified-empty scope must be requested")
        if self.intent is Intent.FIND_CONTROLLED_COMPANIES:
            if self.control_policy is ControlQueryPolicy.NOT_APPLICABLE:
                raise ValueError("control goal signatures require a policy")
        elif self.control_policy is not ControlQueryPolicy.NOT_APPLICABLE:
            raise ValueError("control policy is only valid for control goals")
        if self.result_status is GoalResultStatus.NONEMPTY and not self.result_record_ids:
            raise ValueError("a non-empty goal signature requires result records")
        if self.result_status is not GoalResultStatus.NONEMPTY and self.result_record_ids:
            raise ValueError("an empty goal signature cannot select result records")
        return self


class QuerySignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 5
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
        default="entity-match-v2", min_length=1, max_length=80
    )
    locale: str = "zh-CN"
    goals: list[QueryGoalSignature] = Field(default_factory=list, max_length=100)

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
        if self.goals:
            goal_ids = [goal.goal_id for goal in self.goals]
            if len(goal_ids) != len(set(goal_ids)):
                raise ValueError("query signature goal IDs must be unique")
            known_goal_ids = set(goal_ids)
            if any(
                set(goal.depends_on_goal_ids) - known_goal_ids for goal in self.goals
            ):
                raise ValueError("query signature goal dependency is unknown")
            if len(self.goals) == 1:
                goal = self.goals[0]
                if self.intent is not goal.intent or self.result_merge is not goal.aggregation:
                    raise ValueError("single-goal signature summary must match its goal")
            elif (
                self.intent is not Intent.MULTI_GOAL
                or self.result_merge is not ResultMergeStrategy.NOT_APPLICABLE
            ):
                raise ValueError("multi-goal signatures require aggregate summary values")
        return self


class PlannerAnalysisDecision(BaseModel):
    """Planner stage-one semantics as one or more typed research goals."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Intent = Field(
        description=(
            "按 research_goals 数量汇总，而不是按实体数量汇总：恰好一个 goal 时必须"
            "逐字等于该 goal.intent，即使该 goal 含 2、3、5、10 或更多实体；只有两个"
            "或更多 goal 时使用 multi_goal。"
        )
    )
    entity_references: list[PlannerAnalysisEntityReference] = Field(max_length=100)
    research_goals: list[ResearchGoal] = Field(default_factory=list, max_length=100)
    clarification_question: str | None = Field(default=None, max_length=500)
    query_requires_realtime_data: bool

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_scope(cls, value: Any) -> Any:
        """Accept pre-goal scripted fixtures without exposing legacy fields to OpenAI."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        legacy_fields = {
            "relation_types",
            "raw_relation_types",
            "direction",
            "target_types",
            "requested_attributes",
            "result_merge",
            "control_policy",
        }
        intent_value = normalized.get("intent")
        terminal = intent_value in {
            Intent.CLARIFY,
            Intent.CLARIFY.value,
            Intent.UNSUPPORTED,
            Intent.UNSUPPORTED.value,
        } or bool(normalized.get("query_requires_realtime_data"))
        if "research_goals" not in normalized and not terminal:
            references = normalized.get("entity_references") or []
            subject_indexes = [
                index
                for index, reference in enumerate(references)
                if (reference or {}).get("role", EntityReferenceRole.SUBJECT.value)
                == EntityReferenceRole.SUBJECT.value
            ]
            object_indexes = [
                index
                for index, reference in enumerate(references)
                if (reference or {}).get("role") == EntityReferenceRole.OBJECT.value
            ]
            if not subject_indexes:
                subject_indexes = list(range(len(references)))
            normalized["research_goals"] = [
                {
                    "goal_id": "goal_1",
                    "intent": intent_value,
                    "subject_reference_indexes": subject_indexes,
                    "object_reference_indexes": object_indexes,
                    "relation_types": normalized.get("relation_types", []),
                    "raw_relation_types": normalized.get("raw_relation_types", []),
                    "direction": normalized.get(
                        "direction", ResearchDirection.NOT_APPLICABLE.value
                    ),
                    "target_types": normalized.get("target_types", []),
                    "requested_attributes": normalized.get(
                        "requested_attributes", []
                    ),
                    "aggregation": normalized.get(
                        "result_merge", ResultMergeStrategy.NOT_APPLICABLE.value
                    ),
                    "control_policy": normalized.get(
                        "control_policy", ControlQueryPolicy.NOT_APPLICABLE.value
                    ),
                    "depends_on_goal_ids": [],
                }
            ]
        normalized.setdefault("research_goals", [])
        for field_name in legacy_fields:
            normalized.pop(field_name, None)
        return normalized

    @model_validator(mode="after")
    def coherent_analysis(self) -> PlannerAnalysisDecision:
        terminal = self.intent in {Intent.CLARIFY, Intent.UNSUPPORTED} or (
            self.query_requires_realtime_data
        )
        if self.intent is Intent.CLARIFY:
            if not self.clarification_question:
                raise PydanticCustomError(
                    "clarification_question_required",
                    "clarification_question_required",
                )
        elif self.clarification_question:
            raise PydanticCustomError(
                "clarification_question_forbidden",
                "clarification_question_forbidden",
            )

        if terminal:
            if self.research_goals:
                raise PydanticCustomError(
                    "terminal_research_goals_must_be_empty",
                    "terminal_research_goals_must_be_empty",
                )
            return self

        if not self.entity_references:
            raise PydanticCustomError(
                "entity_references_required",
                "entity_references_required",
            )
        if not self.research_goals:
            raise PydanticCustomError(
                "research_goals_required",
                "research_goals_required",
            )
        goal_ids = [goal.goal_id for goal in self.research_goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise PydanticCustomError(
                "research_goal_ids_must_be_unique",
                "research_goal_ids_must_be_unique",
            )
        known_goal_ids = set(goal_ids)
        goals_by_id = {goal.goal_id: goal for goal in self.research_goals}
        reference_count = len(self.entity_references)
        used_indexes: set[int] = set()
        dependency_graph: dict[str, set[str]] = {}
        for goal in self.research_goals:
            indexes = {
                *goal.subject_reference_indexes,
                *goal.object_reference_indexes,
            }
            if any(index >= reference_count for index in indexes):
                raise PydanticCustomError(
                    "goal_entity_index_out_of_range",
                    "goal_entity_index_out_of_range",
                )
            used_indexes.update(indexes)
            if set(goal.depends_on_goal_ids) - known_goal_ids:
                raise PydanticCustomError(
                    "goal_dependency_unknown",
                    "goal_dependency_unknown",
                )
            dependency_graph[goal.goal_id] = set(goal.depends_on_goal_ids)
        consumed_result_goal_ids = {
            result_goal_id
            for goal in self.research_goals
            for result_goal_id in (
                *goal.subject_result_goal_ids,
                *goal.object_result_goal_ids,
            )
        }
        if any(
            not goals_by_id[result_goal_id].target_types
            for result_goal_id in consumed_result_goal_ids
        ):
            raise PydanticCustomError(
                "consumed_goal_target_types_required",
                "consumed_goal_target_types_required",
            )
        if used_indexes != set(range(reference_count)):
            raise PydanticCustomError(
                "entity_reference_not_used_by_goal",
                "entity_reference_not_used_by_goal",
            )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(goal_id: str) -> None:
            if goal_id in visiting:
                raise PydanticCustomError(
                    "goal_dependency_cycle",
                    "goal_dependency_cycle",
                )
            if goal_id in visited:
                return
            visiting.add(goal_id)
            for dependency in dependency_graph[goal_id]:
                visit(dependency)
            visiting.remove(goal_id)
            visited.add(goal_id)

        for goal_id in goal_ids:
            visit(goal_id)
        if len(self.research_goals) == 1:
            if self.intent is not self.research_goals[0].intent:
                raise PydanticCustomError(
                    "single_goal_intent_mismatch",
                    "single_goal_intent_mismatch",
                )
        elif self.intent is not Intent.MULTI_GOAL:
            raise PydanticCustomError(
                "multi_goal_intent_required",
                "multi_goal_intent_required",
            )
        return self

    @property
    def relation_types(self) -> list[RelationType]:
        return list(dict.fromkeys(item for goal in self.research_goals for item in goal.relation_types))

    @property
    def raw_relation_types(self) -> list[str]:
        return list(dict.fromkeys(item for goal in self.research_goals for item in goal.raw_relation_types))

    @property
    def target_types(self) -> list[NodeType]:
        return list(dict.fromkeys(item for goal in self.research_goals for item in goal.target_types))

    @property
    def requested_attributes(self) -> list[str]:
        return list(dict.fromkeys(item for goal in self.research_goals for item in goal.requested_attributes))

    @property
    def result_merge(self) -> ResultMergeStrategy:
        return (
            self.research_goals[0].aggregation
            if len(self.research_goals) == 1
            else ResultMergeStrategy.NOT_APPLICABLE
        )

    @property
    def direction(self) -> ResearchDirection:
        return (
            self.research_goals[0].direction
            if len(self.research_goals) == 1
            else ResearchDirection.NOT_APPLICABLE
        )

    @property
    def control_policy(self) -> ControlQueryPolicy:
        return (
            self.research_goals[0].control_policy
            if len(self.research_goals) == 1
            else ControlQueryPolicy.NOT_APPLICABLE
        )


class PlannerTaskDecision(BaseModel):
    """Planner stage-two output containing only the executable task DAG."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    research_tasks: list[PlannerTaskDraft] = Field(min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_tasks(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("research_tasks"), list):
            return value
        tasks: list[Any] = []
        for raw_task in value["research_tasks"]:
            if not isinstance(raw_task, dict):
                tasks.append(raw_task)
                continue
            task = dict(raw_task)
            # Older scripted fixtures carried provider-authored prose here.
            # Keep accepting them while the native output schema stays purely
            # structural and does not expose this redundant field to the model.
            task.pop("goal", None)
            if "scope_source" in task:
                tasks.append(task)
                continue
            tool = task.get("tool")
            relation_values = {
                item.value if isinstance(item, RelationType) else str(item)
                for item in task.get("relation_types", [])
            }
            raw_values = set(task.get("raw_relation_types", []))
            scope_source = TaskScopeSource.NOT_APPLICABLE.value
            goal_id = task.get("goal_id")
            if tool in {ToolName.RELATIONS, ToolName.RELATIONS.value}:
                goal_id = goal_id or "goal_1"
                if relation_values == {RelationType.CONTROLS.value}:
                    scope_source = TaskScopeSource.CONTROL_EXPLICIT.value
                elif relation_values == {
                    RelationType.FOUNDED.value,
                    RelationType.WORKS_AT.value,
                    RelationType.OWNS.value,
                } and raw_values == {
                    "Founder_of",
                    "Co-founder_of",
                    "CEO_of",
                    "Chairman_of",
                    "Chairwoman_of",
                    "Owns",
                }:
                    scope_source = TaskScopeSource.CONTROL_FALLBACK.value
                else:
                    scope_source = TaskScopeSource.GOAL.value
            tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "goal_id": goal_id,
                    "subject_result_goal_ids": task.get(
                        "subject_result_goal_ids", []
                    ),
                    "object_result_goal_ids": task.get(
                        "object_result_goal_ids", []
                    ),
                    "tool": tool,
                    "subject_reference_indexes": task.get(
                        "subject_reference_indexes", []
                    ),
                    "object_reference_indexes": task.get(
                        "object_reference_indexes", []
                    ),
                    "scope_source": scope_source,
                    "depends_on": task.get("depends_on", []),
                }
            )
        return {**value, "research_tasks": tasks}

    @model_validator(mode="after")
    def coherent_task_dag(self) -> PlannerTaskDecision:
        task_ids = [task.task_id for task in self.research_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("research task IDs must be unique")
        known_task_ids = set(task_ids)
        dependency_graph: dict[str, set[str]] = {}
        for task in self.research_tasks:
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
    research_goals: list[ResearchGoal] = Field(default_factory=list, max_length=100)
    research_tasks: list[ResearchTask] = Field(max_length=100)
    result_merge: ResultMergeStrategy
    clarification_question: str | None = Field(max_length=500)
    query_requires_realtime_data: bool

    @model_validator(mode="before")
    @classmethod
    def synthesize_legacy_goal(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("research_goals"):
            return value
        normalized = dict(value)
        intent = normalized.get("intent")
        terminal = intent in {
            Intent.CLARIFY,
            Intent.CLARIFY.value,
            Intent.UNSUPPORTED,
            Intent.UNSUPPORTED.value,
        } or bool(normalized.get("query_requires_realtime_data"))
        tasks = normalized.get("research_tasks") or []
        if terminal or not tasks:
            normalized["research_goals"] = []
            return normalized
        relation_tasks = [
            task
            for task in tasks
            if (task or {}).get("tool") in {ToolName.RELATIONS, ToolName.RELATIONS.value}
        ]
        scoped_tasks = relation_tasks or tasks
        subjects = list(
            dict.fromkeys(
                index
                for task in scoped_tasks
                for index in (task or {}).get("subject_reference_indexes", [])
            )
        )
        objects = list(
            dict.fromkeys(
                index
                for task in scoped_tasks
                for index in (task or {}).get("object_reference_indexes", [])
            )
        )
        first = scoped_tasks[0]
        control_policy = (
            ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS.value
            if intent in {
                Intent.FIND_CONTROLLED_COMPANIES,
                Intent.FIND_CONTROLLED_COMPANIES.value,
            }
            else ControlQueryPolicy.NOT_APPLICABLE.value
        )
        legacy_direction = first.get(
            "direction", ResearchDirection.NOT_APPLICABLE.value
        )
        if not relation_tasks and intent in {
            Intent.FIND_RELATED_COMPANIES,
            Intent.FIND_RELATED_COMPANIES.value,
            Intent.FIND_CONTROLLED_COMPANIES,
            Intent.FIND_CONTROLLED_COMPANIES.value,
        }:
            # Compatibility for isolated Researcher entity-resolution tests. A
            # production Planner still has to author the relation task explicitly.
            legacy_direction = ResearchDirection.ANY.value
        normalized["research_goals"] = [
            {
                "goal_id": "goal_1",
                "intent": intent,
                "subject_reference_indexes": subjects,
                "object_reference_indexes": objects,
                "relation_types": (
                    [RelationType.CONTROLS.value]
                    if control_policy
                    == ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS.value
                    else first.get("relation_types", [])
                ),
                "raw_relation_types": (
                    []
                    if control_policy
                    == ControlQueryPolicy.EXPLICIT_THEN_STRONG_ASSOCIATIONS.value
                    else first.get("raw_relation_types", [])
                ),
                "direction": legacy_direction,
                "target_types": first.get("target_types", []),
                "requested_attributes": first.get("requested_attributes", []),
                "aggregation": normalized.get(
                    "result_merge", ResultMergeStrategy.NOT_APPLICABLE.value
                ),
                "control_policy": control_policy,
                "depends_on_goal_ids": [],
            }
        ]
        upgraded_tasks = []
        for task in tasks:
            upgraded_tasks.append({"goal_id": "goal_1", **task})
        normalized["research_tasks"] = upgraded_tasks
        return normalized

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
        if terminal and self.research_goals:
            raise ValueError("terminal Planner decisions cannot include research goals")
        if not terminal and not self.research_goals:
            raise ValueError("an executable Planner decision requires research goals")

        goal_ids = [goal.goal_id for goal in self.research_goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("research goal IDs must be unique")
        if len(self.research_goals) == 1:
            goal = self.research_goals[0]
            if self.intent is not goal.intent or self.result_merge is not goal.aggregation:
                raise ValueError("legacy Planner summary must match the single goal")
        elif self.research_goals and (
            self.intent is not Intent.MULTI_GOAL
            or self.result_merge is not ResultMergeStrategy.NOT_APPLICABLE
        ):
            raise ValueError("multiple goals require multi_goal/not_applicable summary")

        task_ids = [task.task_id for task in self.research_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("research task IDs must be unique")
        known_task_ids = set(task_ids)
        reference_count = len(self.entity_references)
        dependency_graph: dict[str, set[str]] = {}
        for task in self.research_tasks:
            if task.goal_id is not None and task.goal_id not in set(goal_ids):
                raise ValueError("research task references an unknown goal")
            indexes = {
                *task.subject_reference_indexes,
                *task.object_reference_indexes,
            }
            if any(index >= reference_count for index in indexes):
                raise ValueError("research task references an unknown entity index")
            if set(task.depends_on) - known_task_ids:
                raise ValueError("research task depends on an unknown task")
            task_result_goals = {
                *task.subject_result_goal_ids,
                *task.object_result_goal_ids,
            }
            if task_result_goals - set(goal_ids):
                raise ValueError("research task consumes an unknown goal result")
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
    CROSS_LANGUAGE_EXACT = "cross_language_exact"


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
    # Internal Chroma ownership binding.  It is never part of the public API or
    # trace, but makes a conversation-scoped payload self-validating rather than
    # relying only on caller-selected record IDs.
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def conversation_owner_matches_scope(self) -> CachedPayload:
        if self.cache_scope is CacheScope.CONVERSATION and self.conversation_id is None:
            raise ValueError("conversation-scoped cache payload requires conversation_id")
        if self.cache_scope is CacheScope.CONTEXT_FREE and self.conversation_id is not None:
            raise ValueError("context-free cache payload cannot carry conversation_id")
        return self


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
    # Kept as an empty compatibility field so existing API clients do not break.
    disclaimer: str = ""

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
