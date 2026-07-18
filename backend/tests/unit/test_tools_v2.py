from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import ToolResult
from app.tools import FixtureRepository, ToolRegistry


DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry(FixtureRepository.load(DATA_DIRECTORY))


def test_tool_specs_have_factory_parts_and_closed_openai_schemas(
    registry: ToolRegistry,
) -> None:
    schemas = registry.openai_function_schemas()
    assert {item["name"] for item in schemas} == {
        "persons",
        "companies",
        "relations",
    }
    expected_fields = {
        "persons": {"query", "person_ids", "match_mode", "attributes", "limit"},
        "companies": {
            "query",
            "company_ids",
            "match_mode",
            "attributes",
            "include_headquarters",
            "limit",
        },
        "relations": {
            "subject_ids",
            "object_ids",
            "direction",
            "relation_types",
            "raw_relation_types",
            "include_endpoints",
            "limit",
        },
    }
    for spec, schema in zip(registry.specs, schemas, strict=True):
        assert spec.description
        assert callable(spec.handler)
        assert callable(spec.result_adapter)
        assert spec.request_model.model_config["extra"] == "forbid"
        assert schema == spec.openai_function_schema()
        assert schema["type"] == "function"
        assert schema["strict"] is True
        parameters = schema["parameters"]
        assert set(parameters["properties"]) == expected_fields[schema["name"]]
        assert set(parameters["required"]) == expected_fields[schema["name"]]
        assert parameters["additionalProperties"] is False
        serialized = json.dumps(parameters)
        assert '"default"' not in serialized
        assert '"$ref"' not in serialized
        assert '"$defs"' not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Mask", "Elun Mask"])
async def test_fuzzy_person_lookup_accepts_elon_without_confusing_mark(
    registry: ToolRegistry, query: str
) -> None:
    result = await registry.persons({"query": query, "match_mode": "fuzzy"})
    assert result.success
    assert [record["id"] for record in result.records] == ["person:P001"]
    assert result.meta.match_proofs[0].score >= 0.75
    assert result.meta.ambiguous is False


@pytest.mark.asyncio
async def test_tools_do_not_translate_or_add_cross_script_aliases(
    registry: ToolRegistry,
) -> None:
    paths = sorted(DATA_DIRECTORY.glob("*.json"))
    before = {path.name: path.read_bytes() for path in paths}
    person = await registry.persons({"query": "马斯克", "match_mode": "fuzzy"})
    company = await registry.companies({"query": "特斯拉", "match_mode": "fuzzy"})
    assert person.records == []
    assert company.records == []
    assert before == {path.name: path.read_bytes() for path in paths}


@pytest.mark.asyncio
async def test_exact_mode_does_not_silently_fuzzy_match(registry: ToolRegistry) -> None:
    result = await registry.persons({"query": "Elun Mask", "match_mode": "exact"})
    assert result.success
    assert result.records == []


@pytest.mark.asyncio
async def test_company_location_projection_uses_raw_city(registry: ToolRegistry) -> None:
    result = await registry.companies(
        {
            "query": "Tesla, Inc.",
            "attributes": ["source_id", "location_id", "location"],
            "include_headquarters": True,
        }
    )
    company = next(record for record in result.records if record["id"] == "company:C001")
    assert company["properties"]["location_id"] == "location:austin"
    assert any(
        record.get("id") == "location:austin" and record.get("label") == "Austin"
        for record in result.records
    )
    assert any(
        record.get("relation_type") == "headquartered_in"
        for record in result.records
    )


@pytest.mark.asyncio
async def test_relation_filters_are_or_within_lists_and_and_across_lists(
    registry: ToolRegistry,
) -> None:
    result = await registry.relations(
        {
            "subject_ids": ["person:P001", "person:P004"],
            "object_ids": ["company:C002", "company:C005"],
            "relation_types": ["founded", "works_at"],
            "raw_relation_types": ["Founder_of"],
            "direction": "outgoing",
            "include_endpoints": False,
            "limit": 200,
        }
    )
    assert result.success
    assert {
        (record["source"], record["target"])
        for record in result.records
    } == {
        ("person:P001", "company:C002"),
        ("person:P004", "company:C005"),
    }


@pytest.mark.asyncio
async def test_relation_direction_is_relative_to_subject(registry: ToolRegistry) -> None:
    outgoing = await registry.relations(
        {
            "subject_ids": ["company:C001"],
            "object_ids": [],
            "relation_types": ["partner_of"],
            "raw_relation_types": [],
            "direction": "outgoing",
            "include_endpoints": False,
            "limit": 200,
        }
    )
    incoming = await registry.relations(
        {
            "subject_ids": ["company:C001"],
            "object_ids": [],
            "relation_types": ["supplier_to"],
            "raw_relation_types": [],
            "direction": "incoming",
            "include_endpoints": False,
            "limit": 200,
        }
    )
    assert all(record["source"] == "company:C001" for record in outgoing.records)
    assert all(record["target"] == "company:C001" for record in incoming.records)


@pytest.mark.asyncio
async def test_raw_relation_filter_separates_current_and_former_roles(
    registry: ToolRegistry,
) -> None:
    current = await registry.relations(
        {
            "subject_ids": ["person:P001"],
            "object_ids": [],
            "relation_types": ["works_at"],
            "raw_relation_types": ["CEO_of"],
            "direction": "outgoing",
            "include_endpoints": False,
            "limit": 200,
        }
    )
    former = await registry.relations(
        {
            "subject_ids": ["person:P001"],
            "object_ids": [],
            "relation_types": ["works_at"],
            "raw_relation_types": ["Former_CEO_of"],
            "direction": "outgoing",
            "include_endpoints": False,
            "limit": 200,
        }
    )
    assert {record["target"] for record in current.records} == {"company:C001"}
    assert {record["target"] for record in former.records} == {"company:C008"}


@pytest.mark.asyncio
async def test_relation_pagination_reports_total_returned_and_truncated(
    registry: ToolRegistry,
) -> None:
    result = await registry.relations(
        {
            "subject_ids": [],
            "object_ids": [],
            "relation_types": [],
            "raw_relation_types": [],
            "direction": "any",
            "include_endpoints": False,
            "limit": 3,
        }
    )
    assert result.meta.total == 109
    assert result.meta.returned == 3
    assert result.meta.truncated is True


@pytest.mark.asyncio
async def test_ownership_row_and_raw_provenance_are_preserved(
    registry: ToolRegistry,
) -> None:
    result = await registry.relations(
        {
            "subject_ids": ["company:C005"],
            "object_ids": ["company:C023"],
            "relation_types": ["owns"],
            "raw_relation_types": ["Owns"],
            "direction": "outgoing",
            "include_endpoints": True,
            "limit": 200,
        }
    )
    edge = next(record for record in result.records if record["record_kind"] == "relation")
    assert edge["id"] == "relation:raw:0025"
    assert edge["properties"]["raw_relation"] == "Owns"
    assert edge["properties"]["source_file"] == "relations 1.json"
    assert edge["properties"]["source_row"] == 25


@pytest.mark.asyncio
async def test_typed_result_is_shared_tool_result_and_errors_are_closed(
    registry: ToolRegistry,
) -> None:
    success = await registry.persons({"query": "马云"})
    assert isinstance(success, ToolResult)
    invalid = await registry.relations({"subject_ids": ["company:C001"], "unknown": 1})
    assert invalid.success is False
    assert invalid.error is not None
    assert invalid.error.code == "invalid_arguments"
    assert invalid.records == []
    assert invalid.evidence == []
