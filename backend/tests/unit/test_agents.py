from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.agents.graph import route_after_planner
from app.agents.planner import Planner
from app.agents.visualizer import Visualizer
from app.schemas import PlannerDecision, QuerySignature
from app.tools import FixtureRepository, ToolRegistry
from tests.fakes import ScriptedModelClient


DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"


def entity_task(task_id: str, tool: str, index: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "验证实体名称并取得工具记录。",
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


def relation_task(
    task_id: str,
    index: int,
    dependency: str,
    *,
    relation_types: list[str] | None = None,
    target_types: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal": "查询已验证主体的直接关系。",
        "tool": "relations",
        "subject_reference_indexes": [index],
        "object_reference_indexes": [],
        "relation_types": relation_types or [],
        "raw_relation_types": [],
        "direction": "any",
        "target_types": target_types or ["company"],
        "requested_attributes": [],
        "depends_on": [dependency],
    }


def planner_output(
    *,
    references: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    intent: str = "find_related_companies",
    result_merge: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "intent": intent,
        "entity_references": references,
        "research_tasks": tasks,
        "result_merge": result_merge,
        "clarification_question": None,
        "query_requires_realtime_data": False,
    }


def current_reference(
    mention: str,
    entity_type: str,
    canonical_name: str | None,
) -> dict[str, Any]:
    return {
        "mention": mention,
        "canonical_name": canonical_name,
        "source": "current_query",
        "role": "subject",
        "expected_types": [entity_type],
        "context_entity_id": None,
    }


def planner_state(query: str, **updates: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "current_query": query,
        "locale": "zh-CN",
        "recent_turns": [],
        "prior_focus_entity_ids": [],
        "focus_entity_ids": [],
        "resolved_entities": {},
        "route_history": [],
        "agent_steps": [],
        "tool_errors": [],
        "planner_contract_retry_count": 0,
        "replan_count": 0,
    }
    state.update(updates)
    return state


@pytest.mark.asyncio
async def test_planner_accepts_optional_catalog_alignment_and_task_dag() -> None:
    output = planner_output(
        references=[current_reference("未知名字", "person", None)],
        tasks=[
            entity_task("resolve", "persons", 0),
            relation_task("relations", 0, "resolve"),
        ],
    )
    model = ScriptedModelClient({"planner": [output]})
    planner = Planner(
        model,
        entity_catalog=(
            {"name": "Fictional Person", "entity_type": "person"},
        ),
    )

    update = await planner(planner_state("未知名字有哪些公司？"))

    assert update["planner_failed"] is False
    assert update["planner_decision"].entity_references[0].canonical_name is None
    assert [task.tool.value for task in update["planner_decision"].research_tasks] == [
        "persons",
        "relations",
    ]


@pytest.mark.asyncio
async def test_planner_rejects_catalog_name_of_wrong_type_once() -> None:
    output = planner_output(
        references=[current_reference("晨星", "person", "Fictional Company")],
        tasks=[entity_task("resolve", "persons", 0)],
    )
    model = ScriptedModelClient({"planner": [output, output]})
    planner = Planner(
        model,
        entity_catalog=(
            {"name": "Fictional Company", "entity_type": "company"},
        ),
    )
    state = planner_state("晨星有哪些公司？")

    first = await planner(state)
    second = await planner({**state, **first})

    assert first["run_status"] == "running"
    assert first["planner_contract_retry_count"] == 1
    assert second["run_status"] == "failed"
    assert second["planner_contract_retry_count"] == 2


@pytest.mark.asyncio
async def test_planner_allows_explicit_new_entity_and_verified_context_in_one_query() -> None:
    output = planner_output(
        references=[
            current_reference("马云", "person", "马云"),
            {
                "mention": "这些公司",
                "canonical_name": None,
                "source": "conversation_context",
                "role": "object",
                "expected_types": ["company"],
                "context_entity_id": "company:C005",
            },
        ],
        tasks=[
            entity_task("resolve", "persons", 0),
            {
                **relation_task("compare", 0, "resolve"),
                "object_reference_indexes": [1],
            },
        ],
        result_merge="direct",
    )
    model = ScriptedModelClient({"planner": [output]})
    planner = Planner(
        model,
        entity_catalog=({"name": "马云", "entity_type": "person"},),
    )

    update = await planner(
        planner_state(
            "马云和这些公司有什么关系？",
            prior_focus_entity_ids=["company:C005"],
            focus_entity_ids=["company:C005"],
        )
    )
    assert update["planner_failed"] is False
    assert update["planner_decision"].result_merge.value == "direct"


@pytest.mark.asyncio
async def test_planner_context_id_must_come_from_begin_turn_focus_only() -> None:
    output = planner_output(
        references=[
            {
                "mention": "这些公司",
                "canonical_name": None,
                "source": "conversation_context",
                "role": "subject",
                "expected_types": ["company"],
                "context_entity_id": "company:C005",
            }
        ],
        tasks=[
            {
                **relation_task("locations", 0, "unused", target_types=["location"]),
                "depends_on": [],
                "relation_types": ["headquartered_in"],
            }
        ],
        intent="locate_entities",
    )
    model = ScriptedModelClient({"planner": [output]})
    planner = Planner(model)

    update = await planner(
        planner_state(
            "这些公司在哪？",
            prior_focus_entity_ids=[],
            resolved_entities={"旧别名": "company:C005"},
            session_graph={
                "nodes": [{"id": "company:C005", "label": "旧公司", "type": "company"}]
            },
        )
    )
    assert update["planner_failed"] is True
    assert update["planner_contract_retry_count"] == 1


def test_planner_terminal_routes_do_not_invoke_researcher() -> None:
    clarification = PlannerDecision.model_validate(
        {
            "intent": "clarify",
            "entity_references": [],
            "research_tasks": [],
            "result_merge": "not_applicable",
            "clarification_question": "你指的是哪一家同名公司？",
            "query_requires_realtime_data": False,
        }
    )
    unsupported = PlannerDecision.model_validate(
        {
            "intent": "unsupported",
            "entity_references": [],
            "research_tasks": [],
            "result_merge": "not_applicable",
            "clarification_question": None,
            "query_requires_realtime_data": False,
        }
    )
    assert route_after_planner({"planner_decision": clarification}) == "clarify"
    assert route_after_planner({"planner_decision": unsupported}) == "error"


@pytest.mark.asyncio
async def test_clarification_visualizer_adds_no_current_graph_facts() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    question = "你指的是哪一家同名公司？"
    decision = PlannerDecision.model_validate(
        {
            "intent": "clarify",
            "entity_references": [],
            "research_tasks": [],
            "result_merge": "not_applicable",
            "clarification_question": question,
            "query_requires_realtime_data": False,
        }
    )
    model = ScriptedModelClient(
        {"visualizer": [{"answer": question, "answer_record_ids": []}]}
    )
    update = await Visualizer(model, repository.data_version)(
        {
            "current_query": "曙光公司有哪些关系？",
            "locale": "zh-CN",
            "planner_decision": decision,
            "focus_entity_ids": ["company:C005"],
            "research_records": [],
            "selected_record_ids": [],
            "tool_evidence": [],
            "route_history": [],
            "agent_steps": [],
        }
    )
    assert update["answer"].startswith(question)
    assert update["query_result_graph"].nodes == []
    assert update["focus_entity_ids"] == ["company:C005"]
    assert update["turn_focus_entity_ids"] == []


@pytest.mark.asyncio
async def test_verified_no_match_visualizer_keeps_signed_subject_focus() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    person = await registry.persons({"query": "马化腾"})
    person_id = str(person.records[0]["id"])
    signature = QuerySignature(
        intent="find_related_companies",
        subject_ids=[person_id],
        requested_relation_types=["founded"],
        verified_empty_relation_types=["founded"],
        target_types=["company"],
    )
    plan = PlannerDecision.model_validate(
        planner_output(
            references=[current_reference("马化腾", "person", "马化腾")],
            tasks=[entity_task("profile", "persons", 0)],
        )
    )
    model = ScriptedModelClient(
        {
            "visualizer": [
                {
                    "answer": "本次已验证的原始 mock 数据中没有匹配关系。",
                    "answer_record_ids": [],
                }
            ]
        }
    )
    update = await Visualizer(model, repository.data_version)(
        {
            "current_query": "马化腾创办了哪些公司？",
            "locale": "zh-CN",
            "planner_decision": plan,
            "query_signature": signature,
            "no_match": True,
            "research_records": list(person.records),
            "selected_record_ids": [person_id],
            "tool_evidence": list(person.evidence),
            "turn_focus_entity_ids": [person_id],
            "route_history": [],
            "agent_steps": [],
        }
    )
    assert update["run_status"] == "success"
    assert update["query_result_graph"].nodes[0].id == person_id
    assert update["focus_entity_ids"] == [person_id]


@pytest.mark.asyncio
async def test_visualizer_projects_all_verified_relation_endpoints() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    relations = await registry.relations(
        {
            "subject_ids": ["person:P004"],
            "object_ids": [],
            "relation_types": ["founded"],
            "raw_relation_types": [],
            "direction": "any",
            "include_endpoints": True,
            "limit": 200,
        }
    )
    selected_ids = [str(record["id"]) for record in relations.records]
    edge_ids = [
        str(record["id"])
        for record in relations.records
        if record["record_kind"] == "relation"
    ]
    signature = QuerySignature(
        intent="find_related_companies",
        subject_ids=["person:P004"],
        object_ids=["company:C005"],
        relation_types=["founded"],
        requested_relation_types=["founded"],
        effective_relation_types=["founded"],
        target_types=["company"],
    )
    model = ScriptedModelClient(
        {
            "visualizer": [
                {"answer": "演示数据存在创办关系。", "answer_record_ids": edge_ids}
            ]
        }
    )
    update = await Visualizer(model, repository.data_version)(
        {
            "current_query": "人物有哪些公司？",
            "locale": "zh-CN",
            "planner_decision": PlannerDecision.model_validate(
                planner_output(
                    references=[current_reference("人物", "person", None)],
                    tasks=[entity_task("profile", "persons", 0)],
                )
            ),
            "query_signature": signature,
            "research_records": list(relations.records),
            "selected_record_ids": selected_ids,
            "tool_evidence": list(relations.evidence),
            "no_match": False,
            "route_history": [],
            "agent_steps": [],
        }
    )
    graph = update["query_result_graph"]
    assert {node.id for node in graph.nodes} == {"person:P004", "company:C005"}
    assert {edge.id for edge in graph.edges} == set(edge_ids)
    assert update["focus_entity_ids"] == ["company:C005"]


@pytest.mark.asyncio
async def test_visualizer_rejects_model_selected_unknown_record() -> None:
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    person = await registry.persons({"query": "马云"})
    person_id = str(person.records[0]["id"])
    model = ScriptedModelClient(
        {
            "visualizer": [
                {"answer": "错误选择。", "answer_record_ids": ["entity:forged"]}
            ]
        }
    )
    update = await Visualizer(model, repository.data_version)(
        {
            "current_query": "人物资料",
            "locale": "zh-CN",
            "planner_decision": PlannerDecision.model_validate(
                planner_output(
                    references=[current_reference("人物", "person", None)],
                    tasks=[entity_task("profile", "persons", 0)],
                    intent="get_person_profile",
                )
            ),
            "query_signature": QuerySignature(
                intent="get_person_profile",
                subject_ids=[person_id],
                target_types=["person"],
            ),
            "research_records": list(person.records),
            "selected_record_ids": [person_id],
            "tool_evidence": list(person.evidence),
            "route_history": [],
            "agent_steps": [],
            "visualizer_contract_retry_count": 0,
        }
    )
    assert update["run_status"] == "running"
    assert update["query_result_graph"].nodes == []
    assert update["query_result_graph"].edges == []
    assert update["query_result_graph"].evidence == []


def test_production_agent_code_does_not_hardcode_acceptance_entities_or_queries() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path(__file__).parents[2] / "app" / "agents"
        ).glob("*.py")
    )
    for forbidden in (
        "马云有哪些公司",
        "马斯克控制的公司",
        "Tesla, Inc.",
        "SpaceX",
        "阿里巴巴集团",
    ):
        assert forbidden not in source
