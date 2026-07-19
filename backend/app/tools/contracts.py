"""Strict contracts for the three local, read-only Researcher tools.

The request models are the single source for validation and OpenAI function
schemas.  Result records are validated here before they cross the tool/agent
boundary, while remaining dictionaries for the Researcher transcript and graph
projection code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from app.ids import normalize_query
from app.schemas import (
    CompanyAttribute,
    PersonAttribute,
    RelationType,
    ToolError,
    ToolName,
    ToolResult,
)


TOOL_CONTRACT_VERSION = "tools-v5"
ENTITY_MATCH_ALGORITHM_VERSION = "entity-match-v2"
FUZZY_ACCEPT_THRESHOLD = 0.75
FUZZY_MIN_MARGIN = 0.08


class MatchMode(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    CROSS_LANGUAGE_EXACT = "cross_language_exact"


class MatchKind(StrEnum):
    EXPLICIT_ID = "explicit_id"
    EXACT_NAME = "exact_name"
    MENTION = "mention"
    SUBSTRING = "substring"
    FUZZY = "fuzzy"
    CROSS_LANGUAGE_EXACT = "cross_language_exact"


class RelationDirection(StrEnum):
    """Direction relative to ``subject_ids`` (never relative to the caller)."""

    ANY = "any"
    OUTGOING = "outgoing"
    INCOMING = "incoming"


class RawRelationType(StrEnum):
    CEO_OF = "CEO_of"
    CHAIRMAN_OF = "Chairman_of"
    CHAIRWOMAN_OF = "Chairwoman_of"
    FORMER_CEO_OF = "Former_CEO_of"
    FORMER_CHAIRMAN_OF = "Former_Chairman_of"
    FORMER_PRESIDENT_OF = "Former_President_of"
    FOUNDER_OF = "Founder_of"
    CO_FOUNDER_OF = "Co-founder_of"
    HEADQUARTERED_IN = "Headquartered_in"
    OWNS = "Owns"
    PARTNER_WITH = "Partner_with"
    SUPPLIER_TO = "Supplier_to"
    INVESTED_IN = "Invested_in"
    COMPETES_WITH = "Competes_with"
    USES_AI_FROM = "Uses_AI_from"


ConstrainedString = Annotated[str, Field(min_length=1, max_length=200)]


class _RequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def strip_string_list_items(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class CrossLanguageQuery(_RequestBase):
    """One auditable Planner-directory rewrite for an exact raw-name lookup."""

    original_query: ConstrainedString = Field(
        description="用户当前问题中的原始实体提及。",
    )
    rewritten_query: ConstrainedString = Field(
        description="Planner 从动态实体目录逐字选择的标准名。",
    )

    @model_validator(mode="after")
    def rewrite_changes_lookup_text(self) -> CrossLanguageQuery:
        if normalize_query(self.original_query) == normalize_query(
            self.rewritten_query
        ):
            raise ValueError("cross-language rewrite must change the lookup text")
        return self


class PersonsRequest(_RequestBase):
    """Find person records by verified ID/name or opt-in lexical fuzzy search."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="在 person 1.json 中检索的原始用户提及或名称。",
    )
    queries: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "批量检索的原始用户提及或名称；每个 query 独立匹配并返回独立证明。"
        ),
    )
    query_rewrites: list[CrossLanguageQuery] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "仅用于 cross_language_exact 的批量改写；逐项同时保留用户原 mention 与"
            "动态目录标准名。"
        ),
    )
    person_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="原始或带命名空间人物 ID 的 OR 列表。",
    )
    match_mode: MatchMode = Field(
        default=MatchMode.EXACT,
        description=(
            "解析顺序固定为 exact、fuzzy、cross_language_exact；最后一种只能在前两步"
            "完整未命中后使用 query_rewrites。"
        ),
    )
    attributes: list[PersonAttribute] = Field(
        default_factory=lambda: list(PersonAttribute),
        max_length=len(PersonAttribute),
        description="从原始人物记录投影中返回的属性。",
    )
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("person_ids", "attributes", mode="after")
    @classmethod
    def unique_values(cls, values: list[object]) -> list[object]:
        return list(dict.fromkeys(values))

    @field_validator("queries", mode="after")
    @classmethod
    def unique_queries(cls, values: list[str]) -> list[str]:
        return _deduplicate_queries(values)

    @field_validator("query_rewrites", mode="after")
    @classmethod
    def unique_query_rewrites(
        cls, values: list[CrossLanguageQuery]
    ) -> list[CrossLanguageQuery]:
        return _deduplicate_rewrites(values)

    @model_validator(mode="after")
    def validate_query_input(self) -> PersonsRequest:
        selected_inputs = sum(
            (
                self.query is not None,
                bool(self.queries),
                bool(self.query_rewrites),
                bool(self.person_ids),
            )
        )
        if selected_inputs > 1:
            raise ValueError(
                "query, queries, query_rewrites, and person_ids are mutually exclusive"
            )
        if self.match_mode is MatchMode.CROSS_LANGUAGE_EXACT:
            if not self.query_rewrites or self.person_ids:
                raise ValueError(
                    "cross_language_exact requires query_rewrites and no IDs"
                )
        elif self.query_rewrites:
            raise ValueError(
                "query_rewrites require match_mode=cross_language_exact"
            )
        if self.person_ids and self.match_mode is not MatchMode.EXACT:
            raise ValueError("person_ids require match_mode=exact")
        return self

    @property
    def lookup_queries(self) -> tuple[str, ...]:
        if self.query_rewrites:
            return tuple(item.rewritten_query for item in self.query_rewrites)
        if self.queries:
            return tuple(self.queries)
        return (self.query,) if self.query is not None else ()

    @property
    def lookup_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.query_rewrites:
            return tuple(
                (item.original_query, item.rewritten_query)
                for item in self.query_rewrites
            )
        return tuple((query, query) for query in self.lookup_queries)


class CompaniesRequest(_RequestBase):
    """Find company records by verified ID/name or opt-in lexical fuzzy search."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="在 company 1.json 中检索的原始用户提及或名称。",
    )
    queries: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "批量检索的原始用户提及或名称；每个 query 独立匹配并返回独立证明。"
        ),
    )
    query_rewrites: list[CrossLanguageQuery] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "仅用于 cross_language_exact 的批量改写；逐项同时保留用户原 mention 与"
            "动态目录标准名。"
        ),
    )
    company_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="原始或带命名空间企业 ID 的 OR 列表。",
    )
    match_mode: MatchMode = Field(
        default=MatchMode.EXACT,
        description=(
            "解析顺序固定为 exact、fuzzy、cross_language_exact；最后一种只能在前两步"
            "完整未命中后使用 query_rewrites。"
        ),
    )
    attributes: list[CompanyAttribute] = Field(
        default_factory=lambda: [
            CompanyAttribute.SOURCE_ID,
            CompanyAttribute.ALIASES,
            CompanyAttribute.FOUNDED_YEAR,
            CompanyAttribute.LEGAL_REP_ID,
            CompanyAttribute.CITY,
            CompanyAttribute.LOCATION_ID,
            CompanyAttribute.DEMO_DATA,
        ],
        max_length=len(CompanyAttribute),
        description="从原始企业记录投影中返回的属性。",
    )
    include_headquarters: bool = Field(
        default=False,
        description="同时返回已验证的总部关系边与地点端点。",
    )
    limit: int = Field(default=30, ge=1, le=100)

    @field_validator("company_ids", "attributes", mode="after")
    @classmethod
    def unique_values(cls, values: list[object]) -> list[object]:
        return list(dict.fromkeys(values))

    @field_validator("queries", mode="after")
    @classmethod
    def unique_queries(cls, values: list[str]) -> list[str]:
        return _deduplicate_queries(values)

    @field_validator("query_rewrites", mode="after")
    @classmethod
    def unique_query_rewrites(
        cls, values: list[CrossLanguageQuery]
    ) -> list[CrossLanguageQuery]:
        return _deduplicate_rewrites(values)

    @model_validator(mode="after")
    def validate_query_input(self) -> CompaniesRequest:
        selected_inputs = sum(
            (
                self.query is not None,
                bool(self.queries),
                bool(self.query_rewrites),
                bool(self.company_ids),
            )
        )
        if selected_inputs > 1:
            raise ValueError(
                "query, queries, query_rewrites, and company_ids are mutually exclusive"
            )
        if self.match_mode is MatchMode.CROSS_LANGUAGE_EXACT:
            if not self.query_rewrites or self.company_ids:
                raise ValueError(
                    "cross_language_exact requires query_rewrites and no IDs"
                )
        elif self.query_rewrites:
            raise ValueError(
                "query_rewrites require match_mode=cross_language_exact"
            )
        if self.company_ids and self.match_mode is not MatchMode.EXACT:
            raise ValueError("company_ids require match_mode=exact")
        return self

    @property
    def lookup_queries(self) -> tuple[str, ...]:
        if self.query_rewrites:
            return tuple(item.rewritten_query for item in self.query_rewrites)
        if self.queries:
            return tuple(self.queries)
        return (self.query,) if self.query is not None else ()

    @property
    def lookup_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.query_rewrites:
            return tuple(
                (item.original_query, item.rewritten_query)
                for item in self.query_rewrites
            )
        return tuple((query, query) for query in self.lookup_queries)


def _deduplicate_queries(values: list[str]) -> list[str]:
    """Deduplicate transport-equivalent inputs without changing match text."""

    unique: dict[str, str] = {}
    for value in values:
        unique.setdefault(normalize_query(value), value)
    return list(unique.values())


def _deduplicate_rewrites(
    values: list[CrossLanguageQuery],
) -> list[CrossLanguageQuery]:
    unique: dict[str, CrossLanguageQuery] = {}
    for value in values:
        source = normalize_query(value.original_query)
        previous = unique.get(source)
        if previous is not None and normalize_query(
            previous.rewritten_query
        ) != normalize_query(value.rewritten_query):
            raise ValueError(
                "one original query cannot have multiple rewritten queries"
            )
        unique.setdefault(source, value)
    return list(unique.values())


def scope_entity_openai_parameters(
    parameters: dict[str, Any],
    *,
    id_field: Literal["person_ids", "company_ids"],
    match_mode: MatchMode,
    queries: tuple[str, ...] = (),
    query_rewrites: tuple[tuple[str, str], ...] = (),
    entity_ids: tuple[str, ...] = (),
    required_attributes: tuple[str, ...] = (),
) -> None:
    """Close one native entity-tool definition to one executable input branch.

    Pydantic model validators correctly reject mixed name/rewrite/ID requests at
    execution time, but those cross-field validators are not represented in the
    generated JSON Schema.  OpenAI therefore needs a request-local projection
    that makes all inactive transport fields empty and fixes the active phase.
    The projection is generic: callers supply only typed task-contract values;
    it never inspects a user query, entity label, stable ID pattern, or fixture.

    Arrays use the Structured Outputs-supported ``minItems``/``maxItems`` and
    item enums.  Runtime validation remains authoritative and independently
    checks the same task contract before dispatch.
    """

    active_branches = sum(
        (bool(queries), bool(query_rewrites), bool(entity_ids))
    )
    if active_branches != 1:
        raise ValueError("one entity lookup branch must be active")
    if query_rewrites and match_mode is not MatchMode.CROSS_LANGUAGE_EXACT:
        raise ValueError("rewrite branch requires cross_language_exact")
    if not query_rewrites and match_mode is MatchMode.CROSS_LANGUAGE_EXACT:
        raise ValueError("cross_language_exact requires the rewrite branch")
    if entity_ids and match_mode is not MatchMode.EXACT:
        raise ValueError("ID branch requires exact matching")

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("entity tool parameters require object properties")
    required_fields = {
        "query",
        "queries",
        "query_rewrites",
        id_field,
        "match_mode",
        "attributes",
    }
    if not required_fields <= properties.keys():
        raise ValueError("entity tool parameters are missing lookup fields")

    # Native calls always use the batch transport, even for one lookup.  The
    # scalar query remains in the public tool contract for API compatibility but
    # is explicitly null in every Researcher provider phase.
    _scope_scalar_enum(properties["query"], [None])
    _scope_string_array(properties["queries"], list(queries))
    _scope_rewrite_array(properties["query_rewrites"], list(query_rewrites))
    _scope_string_array(properties[id_field], list(entity_ids))
    _scope_scalar_enum(properties["match_mode"], [match_mode.value])
    attribute_schema = properties["attributes"]
    attribute_items = (
        attribute_schema.get("items")
        if isinstance(attribute_schema, dict)
        else None
    )
    allowed_attributes = (
        set(attribute_items.get("enum", []))
        if isinstance(attribute_items, dict)
        else set()
    )
    if set(required_attributes) - allowed_attributes:
        raise ValueError("entity task requests an unsupported tool attribute")
    # Attributes are part of the typed Planner task contract, not a free model
    # preference.  Closing both the cardinality and item vocabulary prevents a
    # native call from silently dropping a property needed by profile evidence.
    _scope_string_array(attribute_schema, list(required_attributes))


def _scope_scalar_enum(schema: Any, values: list[Any]) -> None:
    if not isinstance(schema, dict):
        raise ValueError("phase-scoped scalar must have a JSON Schema")
    schema["enum"] = values


def _scope_string_array(schema: Any, values: list[str]) -> None:
    if not isinstance(schema, dict):
        raise ValueError("phase-scoped list must have a JSON Schema")
    schema["minItems"] = len(values)
    schema["maxItems"] = len(values)
    item_schema = schema.get("items")
    if not isinstance(item_schema, dict):
        raise ValueError("phase-scoped list requires an item schema")
    if values:
        item_schema["enum"] = list(dict.fromkeys(values))
    else:
        item_schema.pop("enum", None)


def _scope_rewrite_array(
    schema: Any,
    values: list[tuple[str, str]],
) -> None:
    if not isinstance(schema, dict):
        raise ValueError("phase-scoped rewrite list must have a JSON Schema")
    schema["minItems"] = len(values)
    schema["maxItems"] = len(values)
    if not values:
        return

    branches = [
        {
            "type": "object",
            "properties": {
                "original_query": {"type": "string", "enum": [original]},
                "rewritten_query": {"type": "string", "enum": [rewritten]},
            },
            "required": ["original_query", "rewritten_query"],
            "additionalProperties": False,
        }
        for original, rewritten in values
    ]
    schema["items"] = branches[0] if len(branches) == 1 else {"anyOf": branches}


class RelationsRequest(_RequestBase):
    """Query raw relation rows using explicit subject-relative semantics.

    Values inside one filter are OR-ed.  Different non-empty filters are AND-ed.
    ``direction`` is always relative to ``subject_ids``: outgoing means subject is
    the raw head, incoming means subject is the raw tail, and any permits either.

    """

    subject_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="已验证主体 ID 的 OR 列表；direction 相对于这些 ID。",
    )
    object_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="已验证对端 ID 的 OR 列表。",
    )
    direction: RelationDirection = Field(
        default=RelationDirection.ANY,
        description="相对于 subject_ids 的关系方向。",
    )
    relation_types: list[RelationType] = Field(
        default_factory=list,
        max_length=len(RelationType),
        description="规范化关系类型的 OR 列表。",
    )
    raw_relation_types: list[RawRelationType] = Field(
        default_factory=list,
        max_length=len(RawRelationType),
        description="原始关系词的精确 OR 列表；与 relation_types 之间采用 AND。",
    )
    include_endpoints: bool = Field(
        default=True,
        description="为每条返回关系同时返回两个端点实体记录。",
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=200,
        description="最大关系行数；结果仍提供 total 与 truncated 元数据。",
    )

    @field_validator(
        "subject_ids",
        "object_ids",
        "relation_types",
        "raw_relation_types",
        mode="after",
    )
    @classmethod
    def unique_values(cls, values: list[object]) -> list[object]:
        return list(dict.fromkeys(values))


class EntityMatchProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str
    query: str
    rewritten_query: str | None = None
    matched_text: str
    kind: MatchKind
    score: float = Field(ge=0.0, le=1.0)
    algorithm: str = ENTITY_MATCH_ALGORITHM_VERSION

    @model_validator(mode="after")
    def validate_rewrite_proof(self) -> EntityMatchProof:
        is_rewrite = self.kind is MatchKind.CROSS_LANGUAGE_EXACT
        if is_rewrite != (self.rewritten_query is not None):
            raise ValueError(
                "cross-language proof and rewritten_query must appear together"
            )
        if self.rewritten_query is not None and normalize_query(
            self.query
        ) == normalize_query(self.rewritten_query):
            raise ValueError("cross-language proof must record a changed query")
        return self


class EntityQueryMatchMeta(BaseModel):
    """Auditable result for one member of a batch entity lookup."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    rewritten_query: str | None = Field(default=None, min_length=1, max_length=200)
    match_mode: MatchMode
    matched_entity_ids: list[str] = Field(default_factory=list, max_length=100)
    match_proofs: list[EntityMatchProof] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    returned: int = Field(default=0, ge=0)
    truncated: bool = False
    ambiguous: bool = False

    @field_validator("matched_entity_ids", mode="after")
    @classmethod
    def unique_entity_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_counts(self) -> EntityQueryMatchMeta:
        if self.returned != len(self.matched_entity_ids):
            raise ValueError("returned must equal matched_entity_ids length")
        if self.returned > self.total:
            raise ValueError("returned cannot exceed total")
        if self.truncated != (self.returned < self.total):
            raise ValueError("truncated must exactly reflect returned < total")
        if any(proof.query != self.query for proof in self.match_proofs):
            raise ValueError("every match proof must belong to this query")
        if any(
            proof.rewritten_query != self.rewritten_query
            for proof in self.match_proofs
        ):
            raise ValueError(
                "every match proof must use this query's audited rewrite"
            )
        if (self.match_mode is MatchMode.CROSS_LANGUAGE_EXACT) != (
            self.rewritten_query is not None
        ):
            raise ValueError(
                "cross_language_exact metadata requires an audited rewritten_query"
            )
        proven_ids = {proof.entity_id for proof in self.match_proofs}
        if not set(self.matched_entity_ids).issubset(proven_ids):
            raise ValueError("every matched entity requires a query-scoped proof")
        return self


class ToolResultMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = TOOL_CONTRACT_VERSION
    total: int = Field(default=0, ge=0)
    returned: int = Field(default=0, ge=0)
    truncated: bool = False
    match_mode: MatchMode | None = None
    match_proofs: list[EntityMatchProof] = Field(default_factory=list)
    query_matches: list[EntityQueryMatchMeta] = Field(default_factory=list)
    ambiguous: bool = False
    acceptance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_margin: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> ToolResultMeta:
        if self.returned > self.total:
            raise ValueError("returned cannot exceed total")
        if self.truncated != (self.returned < self.total):
            raise ValueError("truncated must exactly reflect returned < total")
        return self


class EntityToolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_kind: Literal["entity"]
    id: str
    entity_type: Literal["person", "company", "location"]
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class RelationToolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_kind: Literal["relation"]
    id: str
    source: str
    target: str
    relation_type: RelationType
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


ToolRecord = Annotated[
    EntityToolRecord | RelationToolRecord,
    Field(discriminator="record_kind"),
]
_TOOL_RECORD_ADAPTER = TypeAdapter(ToolRecord)


class TypedToolResult(ToolResult):
    """ToolResult with strict records and audit metadata."""

    meta: ToolResultMeta = Field(default_factory=ToolResultMeta)

    @model_validator(mode="after")
    def validate_result_contract(self) -> TypedToolResult:
        if self.success:
            if self.error is not None:
                raise ValueError("a successful tool result cannot contain an error")
        elif self.error is None:
            raise ValueError("a failed tool result requires a structured error")
        elif self.records or self.evidence:
            raise ValueError("a failed tool result cannot contain records or evidence")

        for record in self.records:
            _TOOL_RECORD_ADAPTER.validate_python(record)
        evidence_ids = {item.id for item in self.evidence}
        for record in self.records:
            if not set(record.get("evidence_ids", ())).issubset(evidence_ids):
                raise ValueError("tool record references evidence absent from the result")
        return self


def failed_tool_result(
    *,
    tool: ToolName,
    provider: str,
    data_version: str,
    elapsed_ms: int,
    error: ToolError,
) -> TypedToolResult:
    return TypedToolResult(
        success=False,
        tool=tool,
        provider=provider,
        data_version=data_version,
        elapsed_ms=elapsed_ms,
        error=error,
        meta=ToolResultMeta(),
    )
