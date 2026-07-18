from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT
from app.schemas import NodeType, RelationType, ToolName
from app.tools import (
    CompaniesRequest,
    CompanyAttribute,
    DataValidationError,
    FixtureRepository,
    PersonsRequest,
    RelationsRequest,
    ToolRegistry,
    companies,
)


DATA_DIRECTORY = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def repository() -> FixtureRepository:
    return FixtureRepository.load(DATA_DIRECTORY)


@pytest.fixture(scope="module")
def registry(repository: FixtureRepository) -> ToolRegistry:
    return ToolRegistry(repository)


def test_repository_projects_every_raw_row_from_only_the_three_source_files(
    repository: FixtureRepository,
) -> None:
    repository.assert_ready()
    assert repository.directory == DATA_DIRECTORY
    assert repository.provider == "local-raw-json-mock"
    assert repository.data_version.startswith("raw-v1-")
    assert repository.manifest["source_files"] == [
        "person 1.json",
        "company 1.json",
        "relations 1.json",
    ]
    assert repository.manifest["source_counts"] == {
        "persons": 20,
        "companies": 30,
        "relations": 109,
    }
    assert len([node for node in repository.nodes if node.type == NodeType.PERSON]) == 20
    # Nine targets exist only as literal references in relations 1.json. They are
    # projected as clearly marked reference nodes instead of being silently dropped.
    assert len([node for node in repository.nodes if node.type == NodeType.COMPANY]) == 39
    assert len([node for node in repository.nodes if node.type == NodeType.LOCATION]) == 17
    assert len(repository.relations) == 109
    assert len(repository.evidence) == len(repository.nodes) + len(repository.relations)

    node_ids = {node.id for node in repository.nodes}
    evidence_ids = {item.id for item in repository.evidence}
    assert all(
        edge.source in node_ids and edge.target in node_ids for edge in repository.relations
    )
    assert all(
        record.evidence_ids and set(record.evidence_ids) <= evidence_ids
        for record in [*repository.nodes, *repository.relations]
    )


def test_no_control_fact_is_invented_from_roles_or_legal_rep(
    repository: FixtureRepository,
) -> None:
    assert not [edge for edge in repository.relations if edge.type == RelationType.CONTROLS]
    ma = repository.nodes_by_id["person:P004"]
    alibaba = repository.nodes_by_id["company:C005"]
    aliyun = repository.nodes_by_id["company:C023"]
    assert ma.label == "马云"
    assert alibaba.properties["legal_rep_id"] == "P004"
    assert aliyun.properties["legal_rep_id"] == "P004"


def test_raw_legal_rep_values_and_unresolved_relation_targets_are_preserved(
    repository: FixtureRepository,
) -> None:
    assert repository.nodes_by_id["company:C025"].properties["legal_rep_id"] == "P021"
    assert repository.nodes_by_id["company:C026"].properties["legal_rep_id"] == "P022"
    assert repository.nodes_by_id["company:C030"].properties["legal_rep_id"] == "P026"
    tiktok = repository.nodes_by_id["company:raw-reference:tiktok"]
    assert tiktok.properties["raw_reference_only"] is True
    assert all(
        edge.source in repository.nodes_by_id and edge.target in repository.nodes_by_id
        for edge in repository.relations
    )


def test_resolution_uses_only_exact_raw_names_and_source_ids(
    repository: FixtureRepository,
) -> None:
    assert repository.resolve_alias("马云", NodeType.PERSON) == "person:P004"
    assert repository.resolve_alias("Elon Musk", NodeType.PERSON) == "person:P001"
    assert repository.resolve_alias("马斯克", NodeType.PERSON) is None
    assert repository.resolve_alias("Tesla, Inc.", NodeType.COMPANY) == "company:C001"
    assert repository.resolve_alias("C002", NodeType.COMPANY) == "company:C002"
    assert repository.resolve_alias("Hangzhou", NodeType.LOCATION) == "location:hangzhou"
    assert repository.find_mentions("请查马云和阿里巴巴集团的关系") == [
        "company:C005",
        "person:P004",
    ]


def test_curated_file_is_not_a_runtime_source(
    repository: FixtureRepository,
    tmp_path: Path,
) -> None:
    for file_name in repository.manifest["source_files"]:
        shutil.copyfile(DATA_DIRECTORY / file_name, tmp_path / file_name)
    (tmp_path / "curated relations.json").write_text(
        json.dumps([{"relation_type": "controls"}]), encoding="utf-8"
    )
    copied = FixtureRepository.load(tmp_path)
    assert copied.data_version == repository.data_version
    assert copied.manifest["source_files"] == repository.manifest["source_files"]
    assert not [edge for edge in copied.relations if edge.type == RelationType.CONTROLS]


def test_invalid_raw_schema_is_rejected(tmp_path: Path) -> None:
    for file_name in ("person 1.json", "company 1.json", "relations 1.json"):
        shutil.copyfile(DATA_DIRECTORY / file_name, tmp_path / file_name)
    people = json.loads((tmp_path / "person 1.json").read_text(encoding="utf-8"))
    del people[0]["name"]
    (tmp_path / "person 1.json").write_text(json.dumps(people), encoding="utf-8")
    with pytest.raises(DataValidationError, match="invalid raw mock-data schema"):
        FixtureRepository.load(tmp_path)


@pytest.mark.asyncio
async def test_persons_tool_finds_raw_name_in_natural_language(
    registry: ToolRegistry,
) -> None:
    result = await registry.persons({"query": "请查询马云的资料"})
    assert result.success is True
    assert result.tool == ToolName.PERSONS
    assert result.provider == "local-raw-json-mock"
    assert [record["id"] for record in result.records] == ["person:P004"]
    assert result.records[0]["record_kind"] == "entity"
    assert result.records[0]["entity_type"] == "person"
    assert result.evidence[0].id in result.records[0]["evidence_ids"]


@pytest.mark.asyncio
async def test_persons_tool_honors_typed_attribute_projection(
    registry: ToolRegistry,
) -> None:
    request = PersonsRequest(
        person_ids=["person:P004"], attributes=["nationality"], limit=1
    )
    result = await registry.persons(request)
    assert result.success
    assert result.records[0]["properties"] == {
        "source_id": "P004",
        "nationality": "China",
        "source_file": "person 1.json",
        "demo_data": True,
    }


@pytest.mark.asyncio
async def test_companies_tool_returns_verified_headquarters_delta(
    registry: ToolRegistry,
) -> None:
    result = await registry.companies(
        {
            "query": "阿里巴巴集团",
            "attributes": [CompanyAttribute.LOCATION],
        }
    )
    assert result.success
    records = {(record["record_kind"], record["id"]): record for record in result.records}
    assert ("entity", "company:C005") in records
    assert ("entity", "location:hangzhou") in records
    headquarters = records[
        (
            "relation",
            "relation:raw:0064",
        )
    ]
    assert headquarters["source"] == "company:C005"
    assert headquarters["target"] == "location:hangzhou"
    assert headquarters["properties"]["raw_relation"] == "Headquartered_in"
    returned_evidence = {item.id for item in result.evidence}
    assert all(
        set(record["evidence_ids"]) <= returned_evidence for record in result.records
    )


@pytest.mark.asyncio
async def test_relations_tool_returns_raw_ma_yun_founder_rows_and_all_endpoints(
    registry: ToolRegistry,
) -> None:
    result = await registry.relations(
        {
            "subject_ids": ["person:P004"],
            "relation_types": [RelationType.FOUNDED],
            "direction": "outgoing",
        }
    )
    assert result.success
    edge_records = [
        record for record in result.records if record["record_kind"] == "relation"
    ]
    node_records = {
        record["id"]: record
        for record in result.records
        if record["record_kind"] == "entity"
    }
    assert {edge["id"] for edge in edge_records} == {
        "relation:raw:0006",
        "relation:raw:0106",
    }
    assert {edge["target"] for edge in edge_records} == {"company:C005"}
    assert {edge["properties"]["raw_relation"] for edge in edge_records} == {
        "Founder_of"
    }
    assert set(node_records) == {"person:P004", "company:C005"}
    assert all(
        edge["source"] in node_records and edge["target"] in node_records
        for edge in edge_records
    )


@pytest.mark.asyncio
async def test_control_query_is_verified_empty_but_raw_ownership_is_queryable(
    registry: ToolRegistry,
) -> None:
    controls = await registry.relations(
        {
            "subject_ids": ["person:P004"],
            "relation_types": [RelationType.CONTROLS],
            "direction": "outgoing",
        }
    )
    assert controls.success is True
    assert controls.records == []
    assert controls.evidence == []

    ownership = await registry.relations(
        {
            "subject_ids": ["company:C005"],
            "relation_types": [RelationType.OWNS],
            "direction": "outgoing",
        }
    )
    edge = next(
        record for record in ownership.records if record["record_kind"] == "relation"
    )
    assert edge["source"] == "company:C005"
    assert edge["target"] == "company:C023"
    assert edge["properties"]["raw_relation"] == "Owns"


@pytest.mark.asyncio
async def test_relations_tool_supports_subject_relative_direction(
    registry: ToolRegistry,
) -> None:
    outgoing = await registry.relations(
        RelationsRequest(
            subject_ids=["C001"],
            relation_types=[RelationType.HEADQUARTERED_IN],
            direction="outgoing",
            include_endpoints=False,
        )
    )
    assert outgoing.success
    assert len(outgoing.records) == 1
    assert outgoing.records[0]["target"] == "location:austin"

    incoming = await registry.relations(
        {
            "subject_ids": ["location:austin"],
            "relation_types": ["headquartered_in"],
            "direction": "incoming",
            "include_endpoints": False,
        }
    )
    assert [record["source"] for record in incoming.records] == ["company:C001"]


@pytest.mark.asyncio
async def test_invalid_tool_arguments_return_a_structured_error(
    registry: ToolRegistry,
) -> None:
    result = await registry.execute(
        ToolName.COMPANIES,
        {"query": "Tesla", "attributes": ["unrestricted-model-string"]},
    )
    assert result.success is False
    assert result.records == []
    assert result.evidence == []
    assert result.error is not None
    assert result.error.tool == ToolName.COMPANIES
    assert result.error.code == "invalid_arguments"
    assert result.error.retryable is False


def test_request_models_forbid_uncontracted_actions() -> None:
    with pytest.raises(ValidationError):
        CompaniesRequest.model_validate({"action": "delete_all"})


@pytest.mark.asyncio
async def test_empty_lookup_is_a_successful_verified_empty_result(
    registry: ToolRegistry,
) -> None:
    result = await registry.companies({"query": "不存在的企业名称"})
    assert result.success is True
    assert result.records == []
    assert result.evidence == []


@pytest.mark.asyncio
async def test_unknown_explicit_entity_ids_do_not_widen_to_all_entities(
    registry: ToolRegistry,
) -> None:
    company_result = await registry.companies(
        {"company_ids": ["company:does-not-exist"]}
    )
    person_result = await registry.persons(
        {"person_ids": ["person:does-not-exist"]}
    )

    for result in (company_result, person_result):
        assert result.success is True
        assert result.records == []
        assert result.evidence == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"subject_ids": ["person:does-not-exist"]},
        {"subject_ids": ["company:does-not-exist"]},
    ],
)
async def test_unknown_explicit_relation_ids_do_not_widen_to_all_relations(
    registry: ToolRegistry,
    arguments: dict[str, list[str]],
) -> None:
    result = await registry.relations(arguments)

    assert result.success is True
    assert result.records == []
    assert result.evidence == []


@pytest.mark.asyncio
async def test_omitted_id_filters_preserve_unfiltered_lookup(
    registry: ToolRegistry,
) -> None:
    companies_result = await registry.companies({"limit": 1})
    persons_result = await registry.persons({"limit": 1})
    relations_result = await registry.relations(
        {"limit": 1, "include_endpoints": False}
    )

    assert len(companies_result.records) == 1
    assert len(persons_result.records) == 1
    assert len(relations_result.records) == 1


@pytest.mark.asyncio
async def test_public_companies_function_and_registry_dispatch(
    repository: FixtureRepository,
    registry: ToolRegistry,
) -> None:
    direct = await companies({"company_ids": ["C002"]}, repository)
    dispatched = await registry.execute(ToolName.COMPANIES, {"company_ids": ["C002"]})
    assert direct.model_dump() == dispatched.model_dump()

    with pytest.raises(ValueError, match="unsupported Researcher tool"):
        await registry.execute("filesystem", {})
