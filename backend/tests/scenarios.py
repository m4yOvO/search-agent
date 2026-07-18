"""Script builders for deterministic end-to-end StateGraph tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas import PlannerDecision, RelationType
from app.tools import FixtureRepository, ToolRegistry
from tests.fakes import ScriptedModelClient


def _ids(records: list[dict[str, Any]], kind: str) -> list[str]:
    return sorted(
        str(record["id"])
        for record in records
        if record.get("record_kind") == kind
    )


def _entity_task(task_id: str, tool: str, index: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "验证实体名称并取得稳定工具记录。",
        "tool": tool,
        "subject_reference_indexes": [index],
        "object_reference_indexes": [],
        "relation_types": [],
        "raw_relation_types": [],
        "direction": "not_applicable",
        "target_types": [],
        "requested_attributes": [],
        "depends_on": [],
    }


def _relation_task(
    task_id: str,
    indexes: list[int],
    *,
    depends_on: list[str],
    relation_types: list[str],
    target_types: list[str],
    direction: str = "outgoing",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "查询已验证主体的关系记录。",
        "tool": "relations",
        "subject_reference_indexes": indexes,
        "object_reference_indexes": [],
        "relation_types": relation_types,
        "raw_relation_types": [],
        "direction": direction,
        "target_types": target_types,
        "requested_attributes": [],
        "depends_on": depends_on,
    }


def _current_reference(mention: str, canonical: str, entity_type: str) -> dict[str, Any]:
    return {
        "mention": mention,
        "canonical_name": canonical,
        "source": "current_query",
        "role": "subject",
        "expected_types": [entity_type],
        "context_entity_id": None,
    }


def _context_reference(mention: str, entity_id: str, entity_type: str) -> dict[str, Any]:
    return {
        "mention": mention,
        "canonical_name": None,
        "source": "conversation_context",
        "role": "subject",
        "expected_types": [entity_type],
        "context_entity_id": entity_id,
    }


def _plan(
    intent: str,
    references: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> PlannerDecision:
    return PlannerDecision.model_validate(
        {
            "intent": intent,
            "entity_references": references,
            "research_tasks": tasks,
            "result_merge": "not_applicable",
            "clarification_question": None,
            "query_requires_realtime_data": False,
        }
    )


async def build_acceptance_model(
    data_directory: Path,
    *,
    include_followup: bool = True,
    include_related_query: bool = False,
) -> ScriptedModelClient:
    """Build a scripted provider whose facts still come through real mock tools."""

    repository = FixtureRepository.load(data_directory)
    tools = ToolRegistry(repository)

    person = await tools.persons({"query": "马云"})
    person_id = str(person.records[0]["id"])
    founded_arguments = {
        "subject_ids": [person_id],
        "object_ids": [],
        "relation_types": [RelationType.FOUNDED.value],
        "raw_relation_types": [],
        "direction": "outgoing",
        "include_endpoints": True,
        "limit": 200,
    }
    founded = await tools.relations(founded_arguments)
    founded_edge_ids = _ids(founded.records, "relation")
    company_ids = sorted(
        str(record["id"])
        for record in founded.records
        if record.get("record_kind") == "entity"
        and record.get("entity_type") == "company"
    )

    planners: list[Any] = [
        _plan(
            "find_related_companies",
            [_current_reference("马云", "马云", "person")],
            [
                _entity_task("resolve_person", "persons", 0),
                _relation_task(
                    "find_founded",
                    [0],
                    depends_on=["resolve_person"],
                    relation_types=["founded"],
                    target_types=["company"],
                ),
            ],
        )
    ]
    researchers: list[Any] = [
        {"action": "call_tool", "tool": "persons", "arguments": {"query": "马云"}},
        {"action": "call_tool", "tool": "relations", "arguments": founded_arguments},
        {"action": "finish"},
    ]
    visualizers: list[Any] = [
        {
            "answer": "原始演示数据中，该人物与阿里巴巴集团存在创办关系。",
            "answer_record_ids": founded_edge_ids,
        }
    ]

    if include_followup:
        location_arguments = {
            "subject_ids": company_ids,
            "object_ids": [],
            "relation_types": [RelationType.HEADQUARTERED_IN.value],
            "raw_relation_types": [],
            "direction": "outgoing",
            "include_endpoints": True,
            "limit": 200,
        }
        locations = await tools.relations(location_arguments)
        location_edge_ids = _ids(locations.records, "relation")
        planners.append(
            _plan(
                "locate_entities",
                [
                    _context_reference("这些公司", company_id, "company")
                    for company_id in company_ids
                ],
                [
                    _relation_task(
                        "locate",
                        list(range(len(company_ids))),
                        depends_on=[],
                        relation_types=["headquartered_in"],
                        target_types=["location"],
                    )
                ],
            )
        )
        researchers.extend(
            [
                {
                    "action": "call_tool",
                    "tool": "relations",
                    "arguments": location_arguments,
                },
                {"action": "finish"},
            ]
        )
        visualizers.append(
            {
                "answer": "这些公司的演示总部位置已加入图谱。",
                "answer_record_ids": location_edge_ids,
            }
        )

    if include_related_query:
        company = await tools.companies({"query": "阿里巴巴集团"})
        company_id = str(company.records[0]["id"])
        ownership_arguments = {
            "subject_ids": [company_id],
            "object_ids": [],
            "relation_types": [RelationType.OWNS.value],
            "raw_relation_types": [],
            "direction": "outgoing",
            "include_endpoints": True,
            "limit": 200,
        }
        ownership = await tools.relations(ownership_arguments)
        ownership_edge_ids = _ids(ownership.records, "relation")
        planners.append(
            _plan(
                "find_related_companies",
                [_current_reference("阿里巴巴集团", "阿里巴巴集团", "company")],
                [
                    _entity_task("resolve_company", "companies", 0),
                    _relation_task(
                        "find_owned",
                        [0],
                        depends_on=["resolve_company"],
                        relation_types=["owns"],
                        target_types=["company"],
                    ),
                ],
            )
        )
        researchers.extend(
            [
                {
                    "action": "call_tool",
                    "tool": "companies",
                    "arguments": {"query": "阿里巴巴集团"},
                },
                {
                    "action": "call_tool",
                    "tool": "relations",
                    "arguments": ownership_arguments,
                },
                {"action": "finish"},
            ]
        )
        visualizers.append(
            {
                "answer": "原始演示数据中，该企业与阿里云存在持有关系。",
                "answer_record_ids": ownership_edge_ids,
            }
        )

    return ScriptedModelClient(
        {
            "planner": planners,
            "researcher": researchers,
            "visualizer": visualizers,
        }
    )
