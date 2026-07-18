from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.planner import Planner
from app.agents.researcher import Researcher
from app.agents.visualizer import Visualizer
from app.schemas import PlannerDecision, QuerySignature
from app.tools import FixtureRepository, ToolRegistry
from tests.fakes import ScriptedModelClient


DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"


class _UnusedModel:
    pass


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=lambda item: (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else str(item)
            ),
        )
    )


def _person_company_plan() -> PlannerDecision:
    return PlannerDecision.model_validate(
        {
            "intent": "find_related_companies",
            "entity_references": [
                {
                    "mention": "马云",
                    "canonical_name": "马云",
                    "source": "current_query",
                    "role": "subject",
                    "expected_types": ["person"],
                    "context_entity_id": None,
                }
            ],
            "research_tasks": [
                {
                    "task_id": "resolve",
                    "goal": "验证人物名称。",
                    "tool": "persons",
                    "subject_reference_indexes": [0],
                    "object_reference_indexes": [],
                    "relation_types": [],
                    "raw_relation_types": [],
                    "direction": "not_applicable",
                    "target_types": [],
                    "requested_attributes": [],
                    "depends_on": [],
                },
                {
                    "task_id": "relations",
                    "goal": "查询人物的广义企业关联。",
                    "tool": "relations",
                    "subject_reference_indexes": [0],
                    "object_reference_indexes": [],
                    "relation_types": [],
                    "raw_relation_types": [],
                    "direction": "any",
                    "target_types": ["company"],
                    "requested_attributes": [],
                    "depends_on": ["resolve"],
                },
            ],
            "result_merge": "not_applicable",
            "clarification_question": None,
            "query_requires_realtime_data": False,
        }
    )


def _base_research_state(plan: PlannerDecision, data_version: str) -> dict[str, Any]:
    return {
        "current_query": "马云有哪些公司？",
        "locale": "zh-CN",
        "data_version": data_version,
        "planner_decision": plan,
        "research_records": [],
        "tool_evidence": [],
        "tool_errors": [],
        "research_transcript": [],
        "executed_tool_fingerprints": {},
        "research_step_count": 0,
        "tool_call_count": 0,
        "researcher_contract_retry_count": 0,
        "replan_count": 0,
        "route_history": [],
        "agent_steps": [],
        "run_status": "running",
    }


def test_planner_payload_contains_dynamic_catalog_but_no_tool_facts() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    catalog = repository.compact_planner_catalog()
    planner = Planner(
        _UnusedModel(),
        entity_catalog=tuple(catalog["entity_catalog"]),
        raw_relation_vocabulary=tuple(catalog["raw_relation_vocabulary"]),
        available_tools=(
            {"name": "persons", "description": "query people"},
            {"name": "companies", "description": "query companies"},
            {"name": "relations", "description": "query relations"},
        ),
    )
    payload = planner._payload(
        {
            "current_query": "这些公司在哪？",
            "locale": "zh-CN",
            "recent_turns": [],
            "resolved_entities": {"不应复制": "company:C999"},
            "prior_focus_entity_ids": ["company:C001"],
            "session_graph": {
                "nodes": [
                    {"id": "company:C001", "label": "Tesla, Inc.", "type": "company"}
                ]
            },
            "research_records": [{"raw_marker": "must-stay-in-state"}],
            "tool_errors": [],
        },
        replan_count=0,
    )

    assert len(payload["entity_catalog"]) == 50
    assert len(payload["raw_relation_vocabulary"]) == 15
    assert payload["prior_focus_entities"] == [
        {
            "entity_id": "company:C001",
            "name": "Tesla, Inc.",
            "entity_type": "company",
        }
    ]
    assert {
        "resolved_entities",
        "research_records",
        "verified_mock_records",
        "output_contract",
    }.isdisjoint(payload)


@pytest.mark.asyncio
async def test_researcher_payload_advances_from_entity_to_relation_task() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    model = ScriptedModelClient(
        {
            "researcher": [
                {"action": "call_tool", "tool": "persons", "arguments": {"query": "马云"}}
            ]
        }
    )
    researcher = Researcher(model, registry)
    state = _base_research_state(_person_company_plan(), repository.data_version)

    before = researcher._payload(state)
    assert before["task_status"] == [
        {"task_id": "resolve", "status": "ready"},
        {"task_id": "relations", "status": "blocked"},
    ]
    assert before["ready_task_contracts"][0]["tool"] == "persons"

    state = {**state, **(await researcher(state))}
    after = researcher._payload(state)
    assert after["verified_bindings"] == {"0": "person:P004"}
    assert after["task_status"] == [
        {"task_id": "resolve", "status": "completed"},
        {"task_id": "relations", "status": "ready"},
    ]
    contract = after["ready_task_contracts"][0]
    assert contract["tool"] == "relations"
    assert contract["required_arguments"] == {
        "subject_ids": ["person:P004"],
        "object_ids": [],
        "relation_types": [],
        "raw_relation_types": [],
        "direction": "any",
        "include_endpoints": True,
        "limit": 200,
    }
    assert set(after["verified_receipts"][0]) == {
        "task_ids",
        "tool",
        "success",
        "executed",
        "record_ids",
        "returned",
        "truncated",
        "error_code",
    }
    assert {"research_records", "tool_evidence", "research_transcript"}.isdisjoint(after)


def test_researcher_payload_is_materially_smaller_than_internal_state() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    state = _base_research_state(_person_company_plan(), repository.data_version)
    state["research_records"] = [
        {"id": f"entity:{index}", "record_kind": "entity", "large": "x" * 500}
        for index in range(20)
    ]
    payload = Researcher(_UnusedModel(), ToolRegistry(repository))._payload(state)
    assert _json_size(payload) < _json_size(state) * 0.5


@pytest.mark.asyncio
async def test_visualizer_payload_contains_only_selected_verified_records() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    person = await registry.persons({"query": "Elon Musk", "match_mode": "exact"})
    selected_id = str(person.records[0]["id"])
    state = {
        "current_query": "Show the person profile",
        "locale": "en-US",
        "planner_decision": PlannerDecision.model_validate(
            {
                "intent": "get_person_profile",
                "entity_references": [
                    {
                        "mention": "the person",
                        "canonical_name": "Elon Musk",
                        "source": "current_query",
                        "role": "subject",
                        "expected_types": ["person"],
                        "context_entity_id": None,
                    }
                ],
                "research_tasks": [
                    {
                        "task_id": "profile",
                        "goal": "查询人物资料。",
                        "tool": "persons",
                        "subject_reference_indexes": [0],
                        "object_reference_indexes": [],
                        "relation_types": [],
                        "raw_relation_types": [],
                        "direction": "not_applicable",
                        "target_types": ["person"],
                        "requested_attributes": [],
                        "depends_on": [],
                    }
                ],
                "result_merge": "not_applicable",
                "clarification_question": None,
                "query_requires_realtime_data": False,
            }
        ),
        "query_signature": QuerySignature(
            intent="get_person_profile",
            subject_ids=[selected_id],
            target_types=["person"],
            locale="en-US",
        ),
        "research_records": list(person.records),
        "selected_record_ids": [selected_id],
        "tool_evidence": list(person.evidence),
        "tool_errors": [],
    }
    visualizer = Visualizer(_UnusedModel(), repository.data_version)
    records, _ = visualizer._verified_catalog(state)
    payload = visualizer._payload(state, records)

    assert payload["verified_selected_records"] == records
    assert payload["graph_record_ids"] == [selected_id]
    assert payload["allowed_answer_record_ids"] == [selected_id]
    assert {
        "planner_decision",
        "evidence_catalog",
        "tool_errors",
        "prompt_version",
    }.isdisjoint(payload)


@pytest.mark.asyncio
async def test_visualizer_relational_answer_candidates_exclude_entity_ids() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    result = await ToolRegistry(repository).relations(
        {
            "subject_ids": ["person:P004"],
            "object_ids": [],
            "relation_types": ["founded"],
            "raw_relation_types": ["Founder_of"],
            "direction": "outgoing",
            "include_endpoints": True,
            "limit": 200,
        }
    )
    selected_ids = [str(record["id"]) for record in result.records]
    state = {
        "current_query": "虚构人物创办了哪些企业？",
        "locale": "zh-CN",
        "query_signature": QuerySignature(
            intent="find_related_companies",
            subject_ids=["person:P004"],
            object_ids=["company:C005"],
            relation_types=["founded"],
            requested_relation_types=["founded"],
            effective_relation_types=["founded"],
            target_types=["company"],
        ),
        "research_records": list(result.records),
        "selected_record_ids": selected_ids,
        "tool_evidence": list(result.evidence),
        "no_match": False,
    }
    visualizer = Visualizer(_UnusedModel(), repository.data_version)
    records, _ = visualizer._verified_catalog(state)
    payload = visualizer._payload(state, records)

    assert set(payload["allowed_answer_record_ids"]) == {
        str(record["id"])
        for record in result.records
        if record["record_kind"] == "relation"
    }
    assert not any(
        record_id.startswith(("person:", "company:"))
        for record_id in payload["allowed_answer_record_ids"]
    )
