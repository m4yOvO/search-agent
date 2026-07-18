from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.agents import AgentDependencies, compile_agent_graph
from app.config import Settings
from app.memory.checkpoint import open_checkpointer
from app.memory.compactor import compact_turns
from app.schemas import (
    CacheScope,
    ConversationSummary,
    ConversationTurn,
    Intent,
    NodeType,
    PlannerDecision,
    QuerySignature,
    RelationType,
    ResearchAction,
    ResearcherDecision,
    ToolName,
)
from app.tools import FixtureRepository, ToolRegistry
from tests.fakes import ScriptedModelClient


DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"


def _turn(number: int) -> ConversationTurn:
    return ConversationTurn(
        user=f"question-{number}",
        assistant=f"answer-{number}",
        focus_entity_ids=[f"company:{number}"],
    )


def test_compacts_oldest_ten_at_fifteen_turn_boundary() -> None:
    retained, summary = compact_turns(
        [_turn(number) for number in range(15)],
        ConversationSummary(),
        focus_entity_ids=["company:current"],
        fact_ids=["company:current", "relation:current-location"],
        latest_graph_id="graph:latest",
    )

    assert [item.user for item in retained] == [f"question-{number}" for number in range(10, 15)]
    assert summary.summarized_turns == 10
    assert summary.user_goals == [f"question-{number}" for number in range(10)]
    assert "company:0" in summary.focus_entity_ids
    assert "company:current" in summary.focus_entity_ids
    assert summary.confirmed_fact_ids == [
        "company:current",
        "relation:current-location",
    ]
    assert summary.latest_graph_id == "graph:latest"


def test_oversized_checkpoint_never_drops_unsummarized_middle_turns() -> None:
    retained, summary = compact_turns(
        [_turn(number) for number in range(17)], ConversationSummary()
    )

    assert [item.user for item in retained] == [f"question-{number}" for number in range(12, 17)]
    assert summary.user_goals == [f"question-{number}" for number in range(12)]
    assert summary.summarized_turns == 12


def test_below_boundary_keeps_raw_turns_but_updates_structured_context() -> None:
    turns = [_turn(number) for number in range(14)]
    retained, summary = compact_turns(
        turns,
        ConversationSummary(),
        resolved_entities={"马斯克": "person:P001"},
        focus_entity_ids=["company:C001"],
    )

    assert retained == turns
    assert summary.summarized_turns == 0
    assert summary.resolved_entities == {"马斯克": "person:P001"}
    assert summary.focus_entity_ids == ["company:C001"]


def test_summary_goals_are_deterministic_and_strictly_bounded() -> None:
    long_query = "   请查询   " + "非常长的企业关系问题 " * 40
    turns = [
        ConversationTurn(
            user=f"{number} {long_query}",
            assistant="answer",
            intent=Intent.FIND_RELATED_COMPANIES,
        )
        for number in range(30)
    ]
    existing = ConversationSummary(
        user_goals=[f"existing-goal-{number}" for number in range(20)]
    )

    _, summary = compact_turns(turns, existing)

    assert len(summary.user_goals) == 20
    assert all(len(goal) <= 120 for goal in summary.user_goals)
    assert all("  " not in goal for goal in summary.user_goals)
    assert all(
        goal.startswith("find_related_companies:") for goal in summary.user_goals
    )


@pytest.mark.asyncio
async def test_async_sqlite_checkpoint_retains_at_most_fifteen_raw_turns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        data_directory=DATA_DIRECTORY,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
    )
    repository = FixtureRepository.load(DATA_DIRECTORY)
    registry = ToolRegistry(repository)
    people = await registry.persons({"query": "马云"})
    person_id = next(record["id"] for record in people.records)
    relation_arguments = {
        "subject_ids": [person_id],
        "object_ids": [],
        "relation_types": ["founded"],
        "raw_relation_types": [],
        "direction": "outgoing",
        "include_endpoints": True,
        "limit": 200,
    }
    relations = await registry.relations(relation_arguments)
    record_map = {
        (record["record_kind"], record["id"]): record
        for record in [*people.records, *relations.records]
    }
    records = list(record_map.values())
    edge_ids = [record["id"] for record in records if record["record_kind"] == "relation"]
    company_ids = [
        record["id"]
        for record in records
        if record["record_kind"] == "entity" and record["entity_type"] == "company"
    ]
    planner_response = PlannerDecision.model_validate(
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
                    "goal": "Resolve the person.",
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
                    "goal": "Retrieve explicit founder records.",
                    "tool": "relations",
                    "subject_reference_indexes": [0],
                    "object_reference_indexes": [],
                    "relation_types": ["founded"],
                    "raw_relation_types": [],
                    "direction": "outgoing",
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
    research_cycle = [
        {"action": "call_tool", "tool": "persons", "arguments": {"query": "马云"}},
        {"action": "call_tool", "tool": "relations", "arguments": relation_arguments},
        {"action": "finish"},
    ]
    visualizer_response = {
        "answer": "演示数据中找到了明确标注的关系。",
        "answer_record_ids": edge_ids,
    }
    model = ScriptedModelClient(
        {
            "planner": [planner_response] * 15,
            "researcher": research_cycle * 15,
            "visualizer": [visualizer_response] * 15,
        }
    )
    config = {
        "configurable": {"thread_id": "compaction-conversation"},
        "recursion_limit": settings.graph_recursion_limit,
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    async with open_checkpointer(settings.checkpoint_path) as checkpointer:
        assert getattr(checkpointer.serde, "_allowed_msgpack_modules", True) is not True
        graph = compile_agent_graph(
            AgentDependencies(
                settings=settings,
                tools=registry,
                cache=None,
                data_version=repository.data_version,
                model=model,
            ),
            checkpointer,
        )
        for turn_number in range(15):
            await graph.ainvoke(
                {
                    "conversation_id": "compaction-conversation",
                    "request_id": f"request-{turn_number}",
                    "current_query": "马云创办了哪些公司？",
                    "locale": "zh-CN",
                },
                config=config,
            )
        snapshot = await graph.aget_state(config)

    assert snapshot.values["total_turn_count"] == 15
    assert len(snapshot.values["recent_turns"]) == 5
    assert snapshot.values["summary"].summarized_turns == 10
    assert not any(
        "Deserializing unregistered type" in record.message
        for record in caplog.records
    )
