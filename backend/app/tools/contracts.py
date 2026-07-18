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

from app.schemas import Evidence, RelationType, ToolError, ToolName, ToolResult


TOOL_CONTRACT_VERSION = "tools-v3"
ENTITY_MATCH_ALGORITHM_VERSION = "entity-match-v1"
FUZZY_ACCEPT_THRESHOLD = 0.75
FUZZY_MIN_MARGIN = 0.08


class MatchMode(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"


class MatchKind(StrEnum):
    EXPLICIT_ID = "explicit_id"
    EXACT_NAME = "exact_name"
    MENTION = "mention"
    SUBSTRING = "substring"
    FUZZY = "fuzzy"


class PersonAttribute(StrEnum):
    SOURCE_ID = "source_id"
    ALIASES = "aliases"
    NATIONALITY = "nationality"
    SUMMARY = "summary"
    DEMO_DATA = "demo_data"


class CompanyAttribute(StrEnum):
    SOURCE_ID = "source_id"
    ALIASES = "aliases"
    FOUNDED_YEAR = "founded_year"
    LEGAL_REP_ID = "legal_rep_id"
    LOCATION_ID = "location_id"
    LOCATION = "location"
    DEMO_DATA = "demo_data"


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


class PersonsRequest(_RequestBase):
    """Find person records by verified ID/name or opt-in lexical fuzzy search."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="在 person 1.json 中检索的原始用户提及或名称。",
    )
    person_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="原始或带命名空间人物 ID 的 OR 列表。",
    )
    match_mode: MatchMode = Field(
        default=MatchMode.EXACT,
        description="先使用 exact；fuzzy 只能作为显式的第二次检索。",
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

class CompaniesRequest(_RequestBase):
    """Find company records by verified ID/name or opt-in lexical fuzzy search."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="在 company 1.json 中检索的原始用户提及或名称。",
    )
    company_ids: list[ConstrainedString] = Field(
        default_factory=list,
        max_length=100,
        description="原始或带命名空间企业 ID 的 OR 列表。",
    )
    match_mode: MatchMode = Field(
        default=MatchMode.EXACT,
        description="先使用 exact；fuzzy 只能作为显式的第二次检索。",
    )
    attributes: list[CompanyAttribute] = Field(
        default_factory=lambda: [
            CompanyAttribute.SOURCE_ID,
            CompanyAttribute.ALIASES,
            CompanyAttribute.FOUNDED_YEAR,
            CompanyAttribute.LEGAL_REP_ID,
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
    matched_text: str
    kind: MatchKind
    score: float = Field(ge=0.0, le=1.0)
    algorithm: str = ENTITY_MATCH_ALGORITHM_VERSION


class ToolResultMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = TOOL_CONTRACT_VERSION
    total: int = Field(default=0, ge=0)
    returned: int = Field(default=0, ge=0)
    truncated: bool = False
    match_mode: MatchMode | None = None
    match_proofs: list[EntityMatchProof] = Field(default_factory=list)
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
