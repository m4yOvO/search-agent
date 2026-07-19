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
    EntityMatchProof,
    EntityQueryMatchMeta,
    MatchKind,
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
    "只在 person 1.json 中检索可验证的人物记录。解析顺序为用户原 mention 的 exact、"
    "原 mention 的 fuzzy，最后才是 Planner 动态目录标准名的 cross_language_exact。"
    "query/queries 保持单项与批量兼容；query_rewrites 为每项保留原 mention 和标准名，"
    "工具逐项返回独立证明。工具只比较原始名称，不翻译、不添加别名，也不猜测 ID。"
)
COMPANIES_DESCRIPTION = (
    "只在 company 1.json 中检索可验证的企业记录，并可按参数返回原始总部关系。解析顺序"
    "为用户原 mention 的 exact、原 mention 的 fuzzy，最后才是 Planner 动态实体目录标准名"
    "的 cross_language_exact。query/queries 保持单项与批量兼容；query_rewrites 为每项保留"
    "原 mention 和标准名并返回独立证明。工具不翻译名称、不添加别名，也不猜测 ID。"
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
        pages = self._entity_search_pages(
            node_type=NodeType.PERSON,
            lookups=request.lookup_pairs,
            entity_ids=request.person_ids,
            match_mode=request.match_mode,
            limit=request.limit,
        )
        attributes = [attribute.value for attribute in request.attributes]
        records = _deduplicate_records(
            self.repository.node_record(node, attributes)
            for _, _, page in pages
            for node in page.nodes
        )
        evidence = self.repository.evidence_for_records(records)
        return ToolHandlerOutput(
            records=tuple(records),
            evidence=tuple(evidence),
            meta=_entity_meta(request.match_mode, pages),
        )

    async def _handle_companies(
        self,
        request: CompaniesRequest,
    ) -> ToolHandlerOutput:
        await asyncio.sleep(0)
        pages = self._entity_search_pages(
            node_type=NodeType.COMPANY,
            lookups=request.lookup_pairs,
            entity_ids=request.company_ids,
            match_mode=request.match_mode,
            limit=request.limit,
        )
        attributes = [
            attribute.value
            for attribute in request.attributes
            if attribute != CompanyAttribute.LOCATION
        ]
        selected_nodes = {
            node.id: node for _, _, page in pages for node in page.nodes
        }
        records = [
            self.repository.node_record(selected_nodes[node_id], attributes)
            for node_id in sorted(selected_nodes)
        ]

        include_headquarters = (
            request.include_headquarters
            or CompanyAttribute.LOCATION in request.attributes
        )
        if include_headquarters and selected_nodes:
            headquarters = self.repository.query_relations(
                subject_ids=selected_nodes,
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
            meta=_entity_meta(request.match_mode, pages),
        )

    def _entity_search_pages(
        self,
        *,
        node_type: NodeType,
        lookups: tuple[tuple[str, str], ...],
        entity_ids: Iterable[str],
        match_mode: MatchMode,
        limit: int,
    ) -> tuple[tuple[str | None, str | None, EntitySearchPage], ...]:
        """Execute each batch member independently against the same ID scope."""

        effective_lookups: tuple[tuple[str | None, str | None], ...] = (
            lookups or ((None, None),)
        )
        repository_mode = (
            MatchMode.EXACT
            if match_mode is MatchMode.CROSS_LANGUAGE_EXACT
            else match_mode
        )
        return tuple(
            (
                original_query,
                effective_query,
                _audited_rewrite_page(
                    self.repository.search_entities_page(
                        node_type=node_type,
                        query=effective_query,
                        entity_ids=entity_ids,
                        match_mode=repository_mode,
                        limit=limit,
                    ),
                    original_query=original_query,
                    rewritten_query=(
                        effective_query
                        if match_mode is MatchMode.CROSS_LANGUAGE_EXACT
                        else None
                    ),
                ),
            )
            for original_query, effective_query in effective_lookups
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
    pages: tuple[tuple[str | None, str | None, EntitySearchPage], ...],
) -> ToolResultMeta:
    total = sum(page.total for _, _, page in pages)
    returned = sum(len(page.nodes) for _, _, page in pages)
    query_matches = [
        EntityQueryMatchMeta(
            query=query,
            rewritten_query=(
                effective_query
                if match_mode is MatchMode.CROSS_LANGUAGE_EXACT
                else None
            ),
            match_mode=match_mode,
            matched_entity_ids=[node.id for node in page.nodes],
            match_proofs=list(page.match_proofs),
            total=page.total,
            returned=len(page.nodes),
            truncated=page.truncated,
            ambiguous=page.ambiguous,
        )
        for query, effective_query, page in pages
        if query is not None
    ]
    return ToolResultMeta(
        total=total,
        returned=returned,
        truncated=returned < total,
        match_mode=match_mode,
        match_proofs=[
            proof for _, _, page in pages for proof in page.match_proofs
        ],
        query_matches=query_matches,
        ambiguous=any(page.ambiguous for _, _, page in pages),
        acceptance_threshold=(
            FUZZY_ACCEPT_THRESHOLD if match_mode is MatchMode.FUZZY else None
        ),
        minimum_margin=FUZZY_MIN_MARGIN if match_mode is MatchMode.FUZZY else None,
    )


def _audited_rewrite_page(
    page: EntitySearchPage,
    *,
    original_query: str | None,
    rewritten_query: str | None,
) -> EntitySearchPage:
    """Attach a non-factual rewrite proof without changing raw matched records."""

    if original_query is None or rewritten_query is None:
        return page
    return EntitySearchPage(
        nodes=page.nodes,
        match_proofs=tuple(
            EntityMatchProof(
                entity_id=proof.entity_id,
                query=original_query,
                rewritten_query=rewritten_query,
                matched_text=proof.matched_text,
                kind=MatchKind.CROSS_LANGUAGE_EXACT,
                score=proof.score,
            )
            for proof in page.match_proofs
        ),
        total=page.total,
        truncated=page.truncated,
        ambiguous=page.ambiguous,
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
