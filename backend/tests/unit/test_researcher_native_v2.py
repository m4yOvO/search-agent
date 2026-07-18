from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.researcher import Researcher
from app.llm import NativeToolCall
from app.schemas import PlannerDecision, ResearchAction, ToolName
from app.tools import FixtureRepository, ToolRegistry
from app.tools.contracts import CompaniesRequest, PersonsRequest


DATA_DIRECTORY = Path(__file__).parents[3] / "data"


def raw_rows(file_name: str) -> list[dict[str, Any]]:
    """Read expected facts from the user-supplied fixture, never from constants."""

    value = json.loads((DATA_DIRECTORY / file_name).read_text(encoding="utf-8"))
    assert isinstance(value, list)
    return value


def raw_entity(
    repository: FixtureRepository,
    *,
    file_name: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    matches = [row for row in raw_rows(file_name) if row.get("name") == name]
    assert len(matches) == 1, f"expected one raw row named {name!r}"
    source_id = str(matches[0]["id"])
    entity_id = repository.canonical_entity_id(source_id)
    assert entity_id is not None
    return matches[0], entity_id


def raw_relation_projection(
    repository: FixtureRepository,
    subject: dict[str, Any],
    *,
    target_type: str,
    direction: str,
    raw_relation_types: set[str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Project the exact raw rows a typed one-hop task is expected to select.

    Returns relation IDs, every closed endpoint ID, and the target-neighbour IDs.
    Row-derived relation IDs deliberately preserve duplicate and label-endpoint rows.
    """

    subject_keys = {str(subject["id"]), str(subject["name"])}
    relation_ids: set[str] = set()
    endpoint_ids: set[str] = set()
    neighbour_ids: set[str] = set()
    for row_number, row in enumerate(raw_rows("relations 1.json"), start=1):
        if raw_relation_types is not None and row["relation"] not in raw_relation_types:
            continue
        head_is_subject = str(row["head"]) in subject_keys
        tail_is_subject = str(row["tail"]) in subject_keys
        candidates: list[str] = []
        if direction in {"outgoing", "any"} and head_is_subject:
            candidates.append(str(row["tail"]))
        if direction in {"incoming", "any"} and tail_is_subject:
            candidates.append(str(row["head"]))
        if not candidates:
            continue
        neighbour_id = next(
            (
                entity_id
                for raw_value in candidates
                if (entity_id := repository.canonical_entity_id(raw_value)) is not None
                and repository.nodes_by_id[entity_id].type.value == target_type
            ),
            None,
        )
        if neighbour_id is None:
            continue
        source_id = repository.canonical_entity_id(str(row["head"]))
        target_id = repository.canonical_entity_id(str(row["tail"]))
        assert source_id is not None and target_id is not None
        relation_ids.add(f"relation:raw:{row_number:04d}")
        endpoint_ids.update({source_id, target_id})
        neighbour_ids.add(neighbour_id)
    return relation_ids, endpoint_ids, neighbour_ids


class NativeScriptedModel:
    provider = "test-openai"
    model_name = "scripted-native"

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.responses = deque(calls)
        self.invocations: list[dict[str, Any]] = []

    async def researcher_tool_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        tools: list[dict[str, Any]],
        purpose: str,
    ) -> NativeToolCall:
        self.invocations.append(
            {
                "prompt": system_prompt,
                "payload": deepcopy(user_payload),
                "tools": deepcopy(tools),
                "purpose": purpose,
            }
        )
        if not self.responses:
            raise AssertionError("no native Researcher response remains")
        return NativeToolCall.model_validate(self.responses.popleft())


class RecordingRegistry:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.calls: list[tuple[ToolName | str, dict[str, Any]]] = []

    @property
    def openai_function_schemas(self):
        return self.registry.openai_function_schemas

    async def execute(self, tool, arguments):
        self.calls.append((tool, deepcopy(arguments)))
        return await self.registry.execute(tool, arguments)


def entity_task(
    task_id: str,
    tool: str,
    reference_index: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "验证目录名称并取得稳定实体记录。",
        "tool": tool,
        "subject_reference_indexes": [reference_index],
        "object_reference_indexes": [],
        "relation_types": [],
        "raw_relation_types": [],
        "direction": "not_applicable",
        "target_types": [],
        "requested_attributes": [],
        "depends_on": [],
    }


def relation_task(
    task_id: str,
    reference_index: int,
    *,
    depends_on: str | list[str],
    relation_types: list[str] | None = None,
    raw_relation_types: list[str] | None = None,
    target_types: list[str] | None = None,
    direction: str = "any",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "查询该主体的直接关系。",
        "tool": "relations",
        "subject_reference_indexes": [reference_index],
        "object_reference_indexes": [],
        "relation_types": relation_types or [],
        "raw_relation_types": raw_relation_types or [],
        "direction": direction,
        "target_types": target_types or ["company"],
        "requested_attributes": [],
        "depends_on": [depends_on] if isinstance(depends_on, str) else depends_on,
    }


def plan(
    references: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    *,
    result_merge: str = "not_applicable",
    intent: str = "find_related_companies",
) -> PlannerDecision:
    return PlannerDecision.model_validate(
        {
            "intent": intent,
            "entity_references": references,
            "research_tasks": tasks,
            "result_merge": result_merge,
            "clarification_question": None,
            "query_requires_realtime_data": False,
        }
    )


def reference(mention: str, canonical_name: str, entity_type: str) -> dict[str, Any]:
    return {
        "mention": mention,
        "canonical_name": canonical_name,
        "source": "current_query",
        "role": "subject",
        "expected_types": [entity_type],
        "context_entity_id": None,
    }


def initial_state(planner: PlannerDecision, data_version: str) -> dict[str, Any]:
    return {
        "planner_decision": planner,
        "current_query": "虚构测试查询",
        "locale": "zh-CN",
        "data_version": data_version,
        "research_records": [],
        "research_transcript": [],
        "tool_evidence": [],
        "tool_errors": [],
        "executed_tool_fingerprints": {},
        "route_history": [],
        "agent_steps": [],
        "research_step_count": 0,
        "tool_call_count": 0,
        "model_call_count": 0,
        "researcher_model_calls": 0,
        "researcher_contract_retry_count": 0,
        "replan_count": 0,
        "run_status": "running",
    }


def person_arguments(query: str) -> dict[str, Any]:
    return PersonsRequest(query=query).model_dump(mode="json")


def relation_arguments(
    subject_id: str | list[str],
    *,
    relation_types: list[str] | None = None,
    raw_relation_types: list[str] | None = None,
    direction: str = "any",
) -> dict[str, Any]:
    return {
        "subject_ids": [subject_id] if isinstance(subject_id, str) else subject_id,
        "object_ids": [],
        "relation_types": relation_types or [],
        "raw_relation_types": raw_relation_types or [],
        "direction": direction,
        "include_endpoints": True,
        "limit": 200,
    }


def company_arguments(query: str) -> dict[str, Any]:
    return CompaniesRequest(query=query).model_dump(mode="json")


async def advance(researcher: Researcher, state: dict[str, Any]) -> dict[str, Any]:
    return {**state, **(await researcher(state))}


@pytest.mark.asyncio
async def test_single_subject_broad_task_uses_native_phases_and_finishes() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    raw_person, person_id = raw_entity(
        repository,
        file_name="person 1.json",
        name="马云",
    )
    expected_edges, expected_endpoints, expected_companies = raw_relation_projection(
        repository,
        raw_person,
        target_type="company",
        direction="any",
    )
    planner = plan(
        [reference("马云", "马云", "person")],
        [
            entity_task("resolve", "persons", 0),
            relation_task("relations", 0, depends_on="resolve"),
        ],
    )
    model = NativeScriptedModel(
        [
            {"name": "persons", "arguments": person_arguments("马云")},
            {
                "name": "relations",
                "arguments": relation_arguments(person_id),
            },
            {"name": "finish", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)

    state = await advance(researcher, state)
    assert [tool["name"] for tool in model.invocations[0]["tools"]] == ["persons"]
    state = await advance(researcher, state)
    assert [tool["name"] for tool in model.invocations[1]["tools"]] == ["relations"]
    state = await advance(researcher, state)
    assert [tool["name"] for tool in model.invocations[2]["tools"]] == ["finish"]

    assert state["run_status"] == "success"
    assert state["query_resolved_entities"] == {"马云": person_id}
    assert set(state["selected_record_ids"]) == {
        person_id,
        *expected_endpoints,
        *expected_edges,
    }
    assert state["query_signature"].relation_types == []
    assert set(state["query_signature"].object_ids) == expected_companies
    assert set(state["turn_focus_entity_ids"]) == expected_companies
    assert len(registry.calls) == 2


@pytest.mark.asyncio
async def test_control_task_dag_preserves_empty_probe_and_raw_fallback_evidence() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    raw_person, person_id = raw_entity(
        repository,
        file_name="person 1.json",
        name="Elon Musk",
    )
    strong_raw_types = {
        "Founder_of",
        "Co-founder_of",
        "CEO_of",
        "Chairman_of",
        "Chairwoman_of",
        "Owns",
    }
    expected_edges, expected_endpoints, expected_companies = raw_relation_projection(
        repository,
        raw_person,
        target_type="company",
        direction="outgoing",
        raw_relation_types=strong_raw_types,
    )
    explicit_task = relation_task(
        "verify_explicit_control",
        0,
        depends_on="resolve_person",
        relation_types=["controls"],
        direction="outgoing",
    )
    fallback_task = relation_task(
        "strong_association_fallback",
        0,
        depends_on="verify_explicit_control",
        relation_types=["founded", "works_at", "owns"],
        raw_relation_types=sorted(strong_raw_types),
        direction="outgoing",
    )
    planner = plan(
        [reference("马斯克", "Elon Musk", "person")],
        [
            entity_task("resolve_person", "persons", 0),
            explicit_task,
            fallback_task,
        ],
        intent="find_controlled_companies",
    )
    model = NativeScriptedModel(
        [
            {"name": "persons", "arguments": person_arguments("Elon Musk")},
            {
                "name": "relations",
                "arguments": relation_arguments(
                    person_id,
                    relation_types=["controls"],
                    direction="outgoing",
                ),
            },
            {
                "name": "relations",
                "arguments": relation_arguments(
                    person_id,
                    relation_types=["founded", "works_at", "owns"],
                    raw_relation_types=sorted(strong_raw_types),
                    direction="outgoing",
                ),
            },
            {"name": "finish", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    for _ in range(4):
        state = await advance(researcher, state)

    assert state["run_status"] == "success"
    assert set(state["selected_record_ids"]) == {
        person_id,
        *expected_endpoints,
        *expected_edges,
    }
    assert set(state["turn_focus_entity_ids"]) == expected_companies

    relation_receipts = [
        receipt
        for receipt in state["research_transcript"]
        if receipt["tool"] == "relations" and receipt["executed"]
    ]
    assert [receipt["task_ids"] for receipt in relation_receipts] == [
        ["verify_explicit_control"],
        ["strong_association_fallback"],
    ]
    assert relation_receipts[0]["record_ids"] == []
    assert relation_receipts[0]["meta"]["truncated"] is False

    signature = state["query_signature"]
    assert {item.value for item in signature.requested_relation_types} == {
        "controls",
        "founded",
        "works_at",
        "owns",
    }
    assert {item.value for item in signature.effective_relation_types} == {
        "founded",
        "works_at",
        "owns",
    }
    assert [item.value for item in signature.verified_empty_relation_types] == [
        "controls"
    ]
    assert signature.control_policy.value == "explicit_then_strong_associations"
    assert set(signature.raw_relation_qualifiers) == strong_raw_types

    selected_records = {
        str(record["id"]): record
        for record in state["research_records"]
        if str(record["id"]) in state["selected_record_ids"]
    }
    selected_relations = [
        record
        for record in selected_records.values()
        if record["record_kind"] == "relation"
    ]
    assert {record["properties"]["raw_relation"] for record in selected_relations} <= (
        strong_raw_types
    )
    assert all(record["relation_type"] != "controls" for record in selected_relations)


@pytest.mark.asyncio
async def test_zero_relation_scope_is_verified_no_results_and_retains_subject() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    _, person_id = raw_entity(
        repository,
        file_name="person 1.json",
        name="马化腾",
    )
    planner = plan(
        [reference("马化腾", "马化腾", "person")],
        [
            entity_task("resolve", "persons", 0),
            relation_task(
                "founded",
                0,
                depends_on="resolve",
                relation_types=["founded"],
                raw_relation_types=["Founder_of", "Co-founder_of"],
            ),
        ],
    )
    model = NativeScriptedModel(
        [
            {"name": "persons", "arguments": person_arguments("马化腾")},
            {
                "name": "relations",
                "arguments": relation_arguments(
                    person_id,
                    relation_types=["founded"],
                    raw_relation_types=["Founder_of", "Co-founder_of"],
                ),
            },
            {"name": "no_results", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    for _ in range(3):
        state = await advance(researcher, state)

    assert state["run_status"] == "success"
    assert state["no_match"] is True
    assert state["selected_record_ids"] == [person_id]
    assert state["turn_focus_entity_ids"] == [person_id]


@pytest.mark.asyncio
async def test_multi_subject_union_is_the_union_of_each_raw_one_hop_scope() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    raw_company, company_id = raw_entity(
        repository,
        file_name="company 1.json",
        name="Tesla, Inc.",
    )
    raw_person, person_id = raw_entity(
        repository,
        file_name="person 1.json",
        name="Elon Musk",
    )
    company_edges, company_endpoints, company_neighbours = raw_relation_projection(
        repository,
        raw_company,
        target_type="company",
        direction="any",
    )
    person_edges, person_endpoints, person_neighbours = raw_relation_projection(
        repository,
        raw_person,
        target_type="company",
        direction="any",
    )
    planner = plan(
        [
            reference("特斯拉", "Tesla, Inc.", "company"),
            reference("马斯克", "Elon Musk", "person"),
        ],
        [
            entity_task("tesla", "companies", 0),
            entity_task("musk", "persons", 1),
            relation_task("tesla_relations", 0, depends_on="tesla"),
            relation_task("musk_relations", 1, depends_on="musk"),
        ],
        result_merge="union",
    )
    model = NativeScriptedModel(
        [
            {"name": "companies", "arguments": company_arguments("Tesla, Inc.")},
            {"name": "persons", "arguments": person_arguments("Elon Musk")},
            {
                "name": "relations",
                "arguments": relation_arguments(company_id),
            },
            {
                "name": "relations",
                "arguments": relation_arguments(person_id),
            },
            {"name": "finish", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    for _ in range(5):
        state = await advance(researcher, state)

    assert state["run_status"] == "success"
    assert set(state["selected_record_ids"]) == {
        company_id,
        person_id,
        *company_edges,
        *person_edges,
        *company_endpoints,
        *person_endpoints,
    }
    assert set(state["query_signature"].subject_ids) == {company_id, person_id}
    assert set(state["query_signature"].object_ids) == {
        *company_neighbours,
        *person_neighbours,
    }
    assert state["query_signature"].result_merge.value == "union"


@pytest.mark.asyncio
async def test_location_followup_uses_only_verified_context_company_ids() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    raw_person, _ = raw_entity(
        repository,
        file_name="person 1.json",
        name="马云",
    )
    _, _, company_ids = raw_relation_projection(
        repository,
        raw_person,
        target_type="company",
        direction="any",
    )
    raw_companies = [
        row
        for row in raw_rows("company 1.json")
        if repository.canonical_entity_id(str(row["id"])) in company_ids
    ]
    expected_edges: set[str] = set()
    expected_endpoints: set[str] = set()
    expected_locations: set[str] = set()
    for raw_company in raw_companies:
        edges, endpoints, locations = raw_relation_projection(
            repository,
            raw_company,
            target_type="location",
            direction="outgoing",
            raw_relation_types={"Headquartered_in"},
        )
        expected_edges.update(edges)
        expected_endpoints.update(endpoints)
        expected_locations.update(locations)

    references = [
        {
            "mention": "这些公司",
            "canonical_name": None,
            "source": "conversation_context",
            "role": "subject",
            "expected_types": ["company"],
            "context_entity_id": company_id,
        }
        for company_id in sorted(company_ids)
    ]
    location_task = {
        "task_id": "locate_context_companies",
        "goal": "查询这些已验证公司的演示总部位置。",
        "tool": "relations",
        "subject_reference_indexes": list(range(len(references))),
        "object_reference_indexes": [],
        "relation_types": ["headquartered_in"],
        "raw_relation_types": [],
        "direction": "outgoing",
        "target_types": ["location"],
        "requested_attributes": [],
        "depends_on": [],
    }
    planner = plan(
        references,
        [location_task],
        intent="locate_entities",
    )
    model = NativeScriptedModel(
        [
            {
                "name": "relations",
                "arguments": relation_arguments(
                    sorted(company_ids),
                    relation_types=["headquartered_in"],
                    direction="outgoing",
                ),
            },
            {"name": "finish", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    for _ in range(2):
        state = await advance(researcher, state)

    assert state["run_status"] == "success"
    assert set(state["selected_record_ids"]) == {
        *expected_edges,
        *expected_endpoints,
    }
    assert set(state["turn_focus_entity_ids"]) == company_ids
    assert set(state["query_signature"].subject_ids) == company_ids
    assert set(state["query_signature"].object_ids) == expected_locations
    assert set(state["query_signature"].context_entity_ids) == company_ids
    assert [tool.value if isinstance(tool, ToolName) else tool for tool, _ in registry.calls] == [
        "relations"
    ]


@pytest.mark.asyncio
async def test_company_target_filter_excludes_locations_and_people() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    _, company_id = raw_entity(
        repository,
        file_name="company 1.json",
        name="Tesla, Inc.",
    )
    planner = plan(
        [reference("特斯拉", "Tesla, Inc.", "company")],
        [
            entity_task("resolve", "companies", 0),
            relation_task("broad", 0, depends_on="resolve"),
        ],
    )
    company_args = company_arguments("Tesla, Inc.")
    model = NativeScriptedModel(
        [
            {"name": "companies", "arguments": company_args},
            {
                "name": "relations",
                "arguments": relation_arguments(company_id),
            },
            {"name": "finish", "arguments": {}},
        ]
    )
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    for _ in range(3):
        state = await advance(researcher, state)

    selected = {
        record["id"]: record
        for record in state["research_records"]
        if record["id"] in state["selected_record_ids"]
    }
    assert all(
        record.get("entity_type") != "location"
        for record in selected.values()
        if record["record_kind"] == "entity"
    )
    assert all(
        record.get("entity_type") != "person"
        for record in selected.values()
        if record["record_kind"] == "entity"
    )
    assert all(
        record.get("relation_type") != "headquartered_in"
        for record in selected.values()
        if record["record_kind"] == "relation"
    )


@pytest.mark.asyncio
async def test_out_of_phase_or_untrusted_relation_call_never_executes() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    planner = plan(
        [reference("未知人物", "Elon Musk", "person")],
        [
            entity_task("resolve", "persons", 0),
            relation_task("relations", 0, depends_on="resolve"),
        ],
    )
    model = NativeScriptedModel(
        [
            {
                "name": "relations",
                "arguments": relation_arguments("person:forged"),
            }
        ]
    )
    state = await advance(
        Researcher(model, registry),
        initial_state(planner, repository.data_version),
    )

    assert registry.calls == []
    assert state["run_status"] == "running"
    assert state["researcher_contract_retry_count"] == 1
    assert state["research_transcript"][-1]["executed"] is False
    assert state["research_transcript"][-1]["error_code"] == "task_scope_mismatch"


@pytest.mark.asyncio
async def test_duplicate_successful_tool_call_is_not_reexecuted() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = RecordingRegistry(ToolRegistry(repository))
    planner = plan(
        [reference("马云", "马云", "person")],
        [entity_task("resolve", "persons", 0)],
    )
    call = {"name": "persons", "arguments": person_arguments("马云")}
    model = NativeScriptedModel([call, call])
    researcher = Researcher(model, registry)
    state = initial_state(planner, repository.data_version)
    state = await advance(researcher, state)

    # Reconstruct a still-pending task while retaining the already executed
    # fingerprint. This isolates the duplicate-execution guard from completion.
    state = {**state, "research_transcript": []}
    from tests.fakes import ScriptedModelClient

    duplicate_model = ScriptedModelClient(
        {
            "researcher": [
                {"action": "call_tool", "tool": "persons", "arguments": person_arguments("马云")}
            ]
        }
    )
    state = await advance(Researcher(duplicate_model, registry), state)
    assert len(registry.calls) == 1
    assert state["needs_replan"] is True
    assert state["research_transcript"][-1]["error_code"] == "duplicate_tool_call"


def test_tool_fingerprint_is_full_sha256_and_stable() -> None:
    arguments = person_arguments("马云")
    first = Researcher._tool_call_fingerprint(ToolName.PERSONS, arguments)
    second = Researcher._tool_call_fingerprint(ToolName.PERSONS, deepcopy(arguments))
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_relation_native_schema_is_scoped_to_all_seven_task_arguments() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    planner = plan(
        [
            {
                "mention": "这些公司",
                "canonical_name": None,
                "source": "conversation_context",
                "role": "subject",
                "expected_types": ["company"],
                "context_entity_id": "company:C005",
            }
        ],
        [
            {
                **relation_task("location", 0, depends_on="placeholder", target_types=["location"]),
                "depends_on": [],
                "relation_types": ["headquartered_in"],
            }
        ],
    )
    researcher = Researcher(NativeScriptedModel([]), registry)
    definitions = researcher._available_fact_tool_definitions(
        initial_state(planner, repository.data_version),
        list(registry.openai_function_schemas()),
    )
    assert [item["name"] for item in definitions] == ["relations"]
    parameters = definitions[0]["parameters"]
    assert set(parameters["required"]) == {
        "subject_ids",
        "object_ids",
        "relation_types",
        "raw_relation_types",
        "direction",
        "include_endpoints",
        "limit",
    }
    assert parameters["properties"]["subject_ids"]["items"]["enum"] == [
        "company:C005"
    ]
