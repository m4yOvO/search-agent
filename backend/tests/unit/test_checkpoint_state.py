from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import aiosqlite
import pytest
from langgraph.channels import UntrackedValue
from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState
from app.memory.checkpoint import CHECKPOINT_ALLOWED_TYPES, open_checkpointer
from app.schemas import (
    CacheMetadata,
    CacheScope,
    ConversationSummary,
    ConversationTurn,
    Evidence,
    GraphNode,
    GraphPayload,
    Intent,
    NodeType,
    PlannerDecision,
    ResearchAction,
    ResearcherDecision,
    ToolName,
    VisualizerDecision,
)


PERSISTED_CHANNELS = {
    "conversation_id",
    "recent_turns",
    "summary",
    "total_turn_count",
    "resolved_entities",
    "focus_entity_ids",
    "latest_graph_id",
    "data_version",
    "session_graph",
}
RAW_SENTINEL = "raw-tool-payload-must-not-be-checkpointed"
QUERY_SENTINEL = "per-request-query-must-not-be-restored"


class _LegacyTrackedState(TypedDict, total=False):
    """Pre-UntrackedValue shape used to exercise checkpoint migration."""

    conversation_id: str
    recent_turns: list[ConversationTurn]
    summary: ConversationSummary
    total_turn_count: int
    resolved_entities: dict[str, str]
    focus_entity_ids: list[str]
    latest_graph_id: str | None
    data_version: str
    session_graph: GraphPayload
    current_query: str
    cache_scope: CacheScope
    planner_decision: PlannerDecision
    researcher_decision: ResearcherDecision
    visualizer_decision: VisualizerDecision
    research_records: list[dict[str, Any]]


def _session_graph() -> GraphPayload:
    evidence = Evidence(
        id="evidence:checkpoint-company",
        provider="checkpoint-test",
        record_id="company:checkpoint",
        source_kind="company",
        updated_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    return GraphPayload(
        graph_id="graph:checkpoint-session",
        nodes=[
            GraphNode(
                id="company:checkpoint",
                type=NodeType.COMPANY,
                label="Checkpoint Company",
                evidence_ids=[evidence.id],
            )
        ],
        evidence=[evidence],
        data_version="checkpoint-v1",
    )


def _state_values() -> dict[str, Any]:
    graph = _session_graph()
    return {
        "conversation_id": "checkpoint-conversation",
        "recent_turns": [
            ConversationTurn(
                user="Where is it?",
                assistant="In the checkpoint fixture.",
                intent=Intent.LOCATE_ENTITIES,
                focus_entity_ids=["company:checkpoint"],
            )
        ],
        "summary": ConversationSummary(
            resolved_entities={"Checkpoint Company": "company:checkpoint"},
            focus_entity_ids=["company:checkpoint"],
            confirmed_fact_ids=["company:checkpoint"],
            confirmed_evidence_ids=["evidence:checkpoint-company"],
            latest_graph_id=graph.graph_id,
        ),
        "total_turn_count": 1,
        "resolved_entities": {"Checkpoint Company": "company:checkpoint"},
        "focus_entity_ids": ["company:checkpoint"],
        "latest_graph_id": graph.graph_id,
        "data_version": graph.data_version,
        "session_graph": graph,
        "cache_scope": CacheScope.CONTEXT_FREE,
        "planner_decision": PlannerDecision.model_validate(
            {
                "intent": "get_company_profile",
                "entity_references": [
                    {
                        "mention": "Checkpoint Company",
                        "canonical_name": "Checkpoint Company",
                        "source": "current_query",
                        "role": "subject",
                        "expected_types": ["company"],
                        "context_entity_id": None,
                    }
                ],
                "research_tasks": [
                    {
                        "task_id": "profile",
                        "goal": "Resolve the company.",
                        "tool": "companies",
                        "subject_reference_indexes": [0],
                        "object_reference_indexes": [],
                        "relation_types": [],
                        "raw_relation_types": [],
                        "direction": "not_applicable",
                        "target_types": ["company"],
                        "requested_attributes": [],
                        "depends_on": [],
                    }
                ],
                "result_merge": "not_applicable",
                "clarification_question": None,
                "query_requires_realtime_data": False,
            }
        ),
        "researcher_decision": ResearcherDecision(
            action=ResearchAction.CALL_TOOL,
            tool=ToolName.COMPANIES,
            arguments={"query": "Checkpoint Company"},
        ),
        "visualizer_decision": VisualizerDecision(
            answer="Verified mock result.",
            answer_record_ids=["company:checkpoint"],
        ),
        "research_records": [
            {
                "record_kind": "entity",
                "id": "company:checkpoint",
                "raw_marker": RAW_SENTINEL,
            }
        ],
        "research_transcript": [{"raw_marker": RAW_SENTINEL}],
        "cache_metadata": CacheMetadata(reason="per-request-only"),
        "answer": "Per-request answer",
        "query_result_graph": graph,
        "graph_id": graph.graph_id,
        "route_history": ["checkpoint-test"],
    }


def _compile_writer(
    state_schema: type[Any], checkpointer: Any, values: dict[str, Any]
) -> Any:
    builder = StateGraph(state_schema)
    builder.add_node("write_state", lambda _state: values)
    builder.add_edge(START, "write_state")
    builder.add_edge("write_state", END)
    return builder.compile(checkpointer=checkpointer)


def _compile_current_reader(checkpointer: Any) -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("noop", lambda _state: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile(checkpointer=checkpointer)


def test_agent_state_persists_only_bounded_conversation_channels() -> None:
    channels = StateGraph(AgentState).channels

    assert {
        name for name, channel in channels.items() if not isinstance(channel, UntrackedValue)
    } == PERSISTED_CHANNELS
    assert isinstance(channels["current_query"], UntrackedValue)
    assert isinstance(channels["research_records"], UntrackedValue)
    assert isinstance(channels["query_resolved_entities"], UntrackedValue)
    assert isinstance(channels["research_transcript"], UntrackedValue)
    assert isinstance(channels["planner_decision"], UntrackedValue)
    assert isinstance(channels["cache_metadata"], UntrackedValue)
    assert isinstance(channels["answer"], UntrackedValue)
    assert isinstance(channels["route_history"], UntrackedValue)
    assert {
        "intent",
        "cache_scope",
        "needs_clarification",
        "clarification_question",
        "query_requires_realtime_data",
        "researcher_decision",
        "tool_data_versions",
        "result_valid",
    }.isdisjoint(channels)


@pytest.mark.asyncio
async def test_sqlite_checkpoint_excludes_transient_payloads_and_restores_session(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "bounded-state.sqlite3"
    config = {"configurable": {"thread_id": "checkpoint-conversation"}}
    values = _state_values()

    async with open_checkpointer(checkpoint_path) as checkpointer:
        graph = _compile_writer(AgentState, checkpointer, values)
        result = await graph.ainvoke(
            {
                "conversation_id": "checkpoint-conversation",
                "request_id": "request-only",
                "current_query": QUERY_SENTINEL,
                "locale": "en-US",
            },
            config=config,
        )
        assert result["research_records"][0]["raw_marker"] == RAW_SENTINEL

    async with open_checkpointer(checkpoint_path) as checkpointer:
        snapshot = await _compile_current_reader(checkpointer).aget_state(config)

    assert set(snapshot.values) == PERSISTED_CHANNELS
    assert snapshot.values["conversation_id"] == "checkpoint-conversation"
    assert snapshot.values["total_turn_count"] == 1
    assert snapshot.values["resolved_entities"] == {
        "Checkpoint Company": "company:checkpoint"
    }
    assert snapshot.values["focus_entity_ids"] == ["company:checkpoint"]
    assert isinstance(snapshot.values["summary"], ConversationSummary)
    assert isinstance(snapshot.values["recent_turns"][0], ConversationTurn)
    assert isinstance(snapshot.values["session_graph"], GraphPayload)

    async with aiosqlite.connect(checkpoint_path) as connection:
        cursor = await connection.execute(
            "SELECT DISTINCT channel FROM writes WHERE thread_id = ?",
            ("checkpoint-conversation",),
        )
        stored_write_channels = {row[0] for row in await cursor.fetchall()}
        cursor = await connection.execute(
            "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = ?",
            ("checkpoint-conversation",),
        )
        checkpoint_rows = await cursor.fetchall()

    assert stored_write_channels.isdisjoint(set(values) - PERSISTED_CHANNELS)
    serialized = b"".join(
        bytes(blob)
        for row in checkpoint_rows
        for blob in row
        if isinstance(blob, (bytes, bytearray, memoryview))
    )
    assert RAW_SENTINEL.encode() not in serialized


@pytest.mark.asyncio
async def test_current_serializer_reads_legacy_agent_decisions_then_drops_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_path = tmp_path / "legacy-state.sqlite3"
    config = {"configurable": {"thread_id": "legacy-conversation"}}
    values = _state_values()
    legacy_values = {
        key: value
        for key, value in values.items()
        if key in _LegacyTrackedState.__annotations__
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    async with open_checkpointer(checkpoint_path) as checkpointer:
        legacy_graph = _compile_writer(_LegacyTrackedState, checkpointer, legacy_values)
        await legacy_graph.ainvoke(legacy_values, config=config)

    async with open_checkpointer(checkpoint_path) as checkpointer:
        snapshot = await _compile_current_reader(checkpointer).aget_state(config)

    assert set(snapshot.values) == PERSISTED_CHANNELS
    assert isinstance(snapshot.values["session_graph"], GraphPayload)
    assert not any(
        "Deserializing unregistered type" in record.message for record in caplog.records
    )
    assert {
        CacheScope,
        PlannerDecision,
        ResearchAction,
        ResearcherDecision,
        VisualizerDecision,
    } <= set(CHECKPOINT_ALLOWED_TYPES)
