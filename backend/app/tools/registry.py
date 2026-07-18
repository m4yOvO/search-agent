"""Reusable async tool registry used by the LangGraph Researcher node."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.schemas import NodeType, RelationType, ToolError, ToolName
from app.tools.contracts import (
    FUZZY_ACCEPT_THRESHOLD,
    FUZZY_MIN_MARGIN,
    CompaniesRequest,
    CompanyAttribute,
    MatchMode,
    PersonsRequest,
    RelationDirection,
    RelationsRequest,
    ToolResultMeta,
    TypedToolResult,
    failed_tool_result,
)
from app.tools.repository import EntitySearchPage, FixtureRepository
from app.tools.specs import ToolHandlerOutput, ToolSpec, create_tool_spec


PERSONS_DESCRIPTION = (
    "只在 person 1.json 中检索可验证的人物记录。先使用 match_mode=exact；仅在精确"
    "检索为空后使用 match_mode=fuzzy。query 可以是 Planner 从动态实体目录选择的标准名；"
    "工具本身只比较原始名称，不翻译、不添加别名，也不猜测 ID。"
)
COMPANIES_DESCRIPTION = (
    "只在 company 1.json 中检索可验证的企业记录，并可按参数返回原始总部关系。先使用"
    " match_mode=exact，仅在精确检索为空后使用 fuzzy。query 可以是 Planner 从动态实体"
    "目录选择的标准名；工具不翻译名称、不添加别名，也不猜测 ID。"
)
RELATIONS_DESCRIPTION = (
    "只查询 relations 1.json，不推断关系。一个 ID 或类型列表内部采用 OR，不同非空过滤"
    "条件之间采用 AND；direction 始终相对于 subject_ids。raw_relation_types 精确过滤"
    "原始关系词，例如区分 Founder_of 与 Former_CEO_of。完成研究前必须检查 "
    "total、returned 和 truncated。"
)
class ToolRegistry:
    """Dispatch the three allow-listed tool specs through one strict boundary."""

    def __init__(self, repository: FixtureRepository) -> None:
        repository.assert_ready()
        self.repository = repository
        self.provider = repository.provider
        self.data_version = repository.data_version
        self._specs: dict[ToolName, ToolSpec[Any]] = {
            ToolName.PERSONS: create_tool_spec(
                name=ToolName.PERSONS,
                description=PERSONS_DESCRIPTION,
                request_model=PersonsRequest,
                handler=self._handle_persons,
                result_adapter=self._adapt_success,
            ),
            ToolName.COMPANIES: create_tool_spec(
                name=ToolName.COMPANIES,
                description=COMPANIES_DESCRIPTION,
                request_model=CompaniesRequest,
                handler=self._handle_companies,
                result_adapter=self._adapt_success,
            ),
            ToolName.RELATIONS: create_tool_spec(
                name=ToolName.RELATIONS,
                description=RELATIONS_DESCRIPTION,
                request_model=RelationsRequest,
                handler=self._handle_relations,
                result_adapter=self._adapt_success,
            ),
        }

    @property
    def specs(self) -> tuple[ToolSpec[Any], ...]:
        return tuple(self._specs[name] for name in ToolName)

    def tool_spec(self, tool: ToolName | str) -> ToolSpec[Any]:
        try:
            return self._specs[ToolName(tool)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unsupported Researcher tool: {tool!r}") from exc

    def openai_function_schemas(self) -> list[dict[str, Any]]:
        """Definitions ready for ``responses.create(tools=...)``."""

        return [spec.openai_function_schema() for spec in self.specs]

    def capability_catalog(self) -> list[dict[str, Any]]:
        """Provider-neutral descriptions generated from the same live specs."""

        return [
            {
                "name": spec.name.value,
                "description": spec.description,
                "parameters": spec.openai_function_schema()["parameters"],
            }
            for spec in self.specs
        ]

    async def execute(
        self,
        tool: ToolName | str,
        arguments: dict[str, Any],
    ) -> TypedToolResult:
        spec = self.tool_spec(tool)
        return await self._invoke(spec, arguments)

    async def persons(
        self,
        request: PersonsRequest | dict[str, Any],
    ) -> TypedToolResult:
        return await self._invoke(self._specs[ToolName.PERSONS], request)

    async def companies(
        self,
        request: CompaniesRequest | dict[str, Any],
    ) -> TypedToolResult:
        return await self._invoke(self._specs[ToolName.COMPANIES], request)

    async def relations(
        self,
        request: RelationsRequest | dict[str, Any],
    ) -> TypedToolResult:
        return await self._invoke(self._specs[ToolName.RELATIONS], request)

    async def _invoke(
        self,
        spec: ToolSpec[Any],
        request: BaseModel | dict[str, Any],
    ) -> TypedToolResult:
        started = time.perf_counter_ns()
        try:
            return await spec.invoke(request, started_ns=started)
        except ValidationError as exc:
            return self._failure(
                spec.name,
                code="invalid_arguments",
                message=_validation_message(exc),
                started=started,
            )
        except Exception as exc:
            return self._failure(
                spec.name,
                code="tool_execution_error",
                message=f"Local mock tool execution failed: {type(exc).__name__}",
                started=started,
            )

    async def _handle_persons(self, request: PersonsRequest) -> ToolHandlerOutput:
        await asyncio.sleep(0)
        page = self.repository.search_entities_page(
            node_type=NodeType.PERSON,
            query=request.query,
            entity_ids=request.person_ids,
            match_mode=request.match_mode,
            limit=request.limit,
        )
        attributes = [attribute.value for attribute in request.attributes]
        records = [
            self.repository.node_record(node, attributes) for node in page.nodes
        ]
        evidence = self.repository.evidence_for_records(records)
        return ToolHandlerOutput(
            records=tuple(records),
            evidence=tuple(evidence),
            meta=_entity_meta(request.match_mode, page),
        )

    async def _handle_companies(
        self,
        request: CompaniesRequest,
    ) -> ToolHandlerOutput:
        await asyncio.sleep(0)
        page = self.repository.search_entities_page(
            node_type=NodeType.COMPANY,
            query=request.query,
            entity_ids=request.company_ids,
            match_mode=request.match_mode,
            limit=request.limit,
        )
        attributes = [
            attribute.value
            for attribute in request.attributes
            if attribute != CompanyAttribute.LOCATION
        ]
        records = [
            self.repository.node_record(node, attributes) for node in page.nodes
        ]

        include_headquarters = (
            request.include_headquarters
            or CompanyAttribute.LOCATION in request.attributes
        )
        if include_headquarters and page.nodes:
            headquarters = self.repository.query_relations(
                subject_ids=[node.id for node in page.nodes],
                relation_types=[RelationType.HEADQUARTERED_IN],
                direction=RelationDirection.OUTGOING,
                limit=200,
            )
            for edge in headquarters.edges:
                records.append(self.repository.relation_record(edge))
                location = self.repository.nodes_by_id[edge.target]
                records.append(self.repository.node_record(location))
        records = _deduplicate_records(records)
        evidence = self.repository.evidence_for_records(records)
        return ToolHandlerOutput(
            records=tuple(records),
            evidence=tuple(evidence),
            meta=_entity_meta(request.match_mode, page),
        )

    async def _handle_relations(self, request: RelationsRequest) -> ToolHandlerOutput:
        await asyncio.sleep(0)
        page = self.repository.query_relations(
            subject_ids=request.subject_ids,
            object_ids=request.object_ids,
            relation_types=request.relation_types,
            raw_relation_types=request.raw_relation_types,
            direction=request.direction,
            limit=request.limit,
        )
        records: list[dict[str, Any]] = []
        for edge in page.edges:
            records.append(self.repository.relation_record(edge))
            if request.include_endpoints:
                records.append(
                    self.repository.node_record(self.repository.nodes_by_id[edge.source])
                )
                records.append(
                    self.repository.node_record(self.repository.nodes_by_id[edge.target])
                )
        records = _deduplicate_records(records)
        evidence = self.repository.evidence_for_records(records)
        return ToolHandlerOutput(
            records=tuple(records),
            evidence=tuple(evidence),
            meta=ToolResultMeta(
                total=page.total,
                returned=len(page.edges),
                truncated=page.truncated,
            ),
        )

    def _adapt_success(
        self,
        tool: ToolName,
        output: ToolHandlerOutput,
        started: int,
    ) -> TypedToolResult:
        return TypedToolResult(
            success=True,
            tool=tool,
            provider=self.provider,
            data_version=self.data_version,
            records=list(output.records),
            evidence=list(output.evidence),
            elapsed_ms=_elapsed_ms(started),
            meta=output.meta,
        )

    def _failure(
        self,
        tool: ToolName,
        *,
        code: str,
        message: str,
        started: int,
    ) -> TypedToolResult:
        return failed_tool_result(
            tool=tool,
            provider=self.provider,
            data_version=self.data_version,
            elapsed_ms=_elapsed_ms(started),
            error=ToolError(
                tool=tool,
                code=code,
                message=message,
                retryable=False,
            ),
        )


def _entity_meta(
    match_mode: MatchMode,
    page: EntitySearchPage,
) -> ToolResultMeta:
    return ToolResultMeta(
        total=page.total,
        returned=len(page.nodes),
        truncated=page.truncated,
        match_mode=match_mode,
        match_proofs=list(page.match_proofs),
        ambiguous=page.ambiguous,
        acceptance_threshold=(
            FUZZY_ACCEPT_THRESHOLD if match_mode is MatchMode.FUZZY else None
        ),
        minimum_margin=FUZZY_MIN_MARGIN if match_mode is MatchMode.FUZZY else None,
    )


def _deduplicate_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["record_kind"]), str(record["id"]))
        deduplicated.setdefault(key, record)
    return list(deduplicated.values())


def _elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)


def _validation_message(error: ValidationError) -> str:
    details = error.errors(include_url=False, include_input=False)
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in details
    )


def build_tool_registry(repository: FixtureRepository) -> ToolRegistry:
    return ToolRegistry(repository)


async def companies(
    request: CompaniesRequest | dict[str, Any],
    repository: FixtureRepository,
) -> TypedToolResult:
    """Public ``companies`` Researcher tool."""

    return await ToolRegistry(repository).companies(request)


async def persons(
    request: PersonsRequest | dict[str, Any],
    repository: FixtureRepository,
) -> TypedToolResult:
    """Public ``persons`` Researcher tool."""

    return await ToolRegistry(repository).persons(request)


async def relations(
    request: RelationsRequest | dict[str, Any],
    repository: FixtureRepository,
) -> TypedToolResult:
    """Public ``relations`` Researcher tool."""

    return await ToolRegistry(repository).relations(request)
