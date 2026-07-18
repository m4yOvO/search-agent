from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import AgentStepTrace, ChatRequest, TraceMetadata
from app.service import ApplicationService, ServiceResources


def _valid_step(**updates: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "role": "researcher",
        "action": "call_tool",
        "tool": "relations",
        "relation_types": ["controls", "founded"],
        "result_merge": None,
        "resolution_strategy": None,
        "resolution_version": None,
        "record_ids": ["person:P001", "relation:raw:0001"],
        "argument_fingerprint": "a" * 64,
        "count": 2,
        "error_code": None,
    }
    step.update(updates)
    return step


def test_agent_step_trace_is_strict_bounded_and_json_safe() -> None:
    step = AgentStepTrace.model_validate(_valid_step())

    assert step.model_dump(mode="json") == _valid_step()
    assert AgentStepTrace.model_validate(
        {
            "role": "visualizer",
            "action": "select_records",
            "record_ids": ["company:raw-reference:比亚迪"],
        }
    ).record_ids == ["company:raw-reference:比亚迪"]
    assert TraceMetadata().agent_steps == []

    planned = AgentStepTrace.model_validate(
        {
            "role": "planner",
            "action": "plan",
            "result_merge": "union",
        }
    )
    assert planned.result_merge == "union"

    resolved = AgentStepTrace.model_validate(
        _valid_step(
            action="tool_result",
            tool="companies",
            relation_types=[],
            resolution_strategy="exact",
            resolution_version="entity-match-v1",
            record_ids=["company:C001"],
        )
    )
    assert resolved.resolution_strategy == "exact"


@pytest.mark.parametrize(
    "updates",
    [
        {"role": "router"},
        {"action": "call tool"},
        {"action": "a" * 65},
        {"tool": "filesystem"},
        {"relation_types": ["controls", "controls"]},
        {"result_merge": "union"},
        {"resolution_strategy": "exact"},
        {"resolution_version": "entity-match-v1"},
        {
            "resolution_strategy": "exact",
            "resolution_version": "entity-match-v1",
        },
        {
            "tool": "companies",
            "resolution_strategy": "exact",
            "resolution_version": "unsafe version with spaces",
        },
        {"record_ids": ["person:P001", "person:P001"]},
        {"record_ids": ["company:../../private-query"]},
        {"record_ids": ["evidence:raw:person:P001"]},
        {"argument_fingerprint": "A" * 64},
        {"argument_fingerprint": "a" * 63},
        {"count": -1},
        {"error_code": "provider reason: raw payload"},
        {"unexpected_payload": "must never be public"},
    ],
)
def test_agent_step_trace_rejects_unsafe_or_unbounded_fields(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        AgentStepTrace.model_validate(_valid_step(**updates))


def test_trace_metadata_limits_agent_steps_to_64() -> None:
    TraceMetadata.model_validate({"agent_steps": [_valid_step()] * 64})

    with pytest.raises(ValidationError):
        TraceMetadata.model_validate({"agent_steps": [_valid_step()] * 65})


class _OneShotGraph:
    def __init__(self, *, include_legacy_trace: bool = False) -> None:
        self.include_legacy_trace = include_legacy_trace

    async def ainvoke(self, input, config):
        result = {
            "answer": "已从本地演示工具获得可验证结果。",
            "session_graph": {
                "graph_id": "graph:trace-contract",
                "nodes": [],
                "edges": [],
                "evidence": [],
                "data_version": "raw-v1",
            },
            "cache_metadata": {},
            "researcher_invoked": True,
            "tool_call_count": 1,
            "research_steps": 2,
            "replan_count": 0,
            "model_provider": "openai",
            "model_name": "test-model",
            "model_call_count": 3,
            "planner_model_calls": 1,
            "researcher_model_calls": 1,
            "visualizer_model_calls": 1,
            "route_history": ["planner", "researcher", "visualizer"],
            "agent_steps": [
                {
                    "role": "planner",
                    "action": "plan",
                    "relation_types": ["founded"],
                    "count": 2,
                },
                _valid_step(),
                {
                    "role": "visualizer",
                    "action": "select_records",
                    "record_ids": ["person:P001", "relation:raw:0001"],
                    "count": 2,
                },
            ],
        }
        if self.include_legacy_trace:
            # Simulate an older aggregate trace assembled before ``agent_steps``.
            result["trace"] = {
                "researcher_invoked": True,
                "tool_calls": 1,
                "research_steps": 2,
                "model_name": "test-model",
                "route_history": ["planner", "researcher", "visualizer"],
            }
        return result


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []

    async def save(self, conversation_id, graph) -> None:
        self.saved.append((conversation_id, graph.graph_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("include_legacy_trace", [False, True])
async def test_service_exposes_state_agent_steps_and_logs_only_counts(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
    include_legacy_trace: bool,
) -> None:
    private_query = "PRIVATE QUERY MUST NOT ENTER CHAT_COMPLETED LOG"
    store = _RecordingGraphStore()
    service = ApplicationService(
        ServiceResources(
            graph=_OneShotGraph(include_legacy_trace=include_legacy_trace),
            graph_store=store,  # type: ignore[arg-type]
            settings=settings,
        )
    )
    caplog.set_level(logging.INFO, logger="app.service")

    response = await service.chat(ChatRequest(message=private_query))

    assert [step.role.value for step in response.trace.agent_steps] == [
        "planner",
        "researcher",
        "visualizer",
    ]
    assert response.trace.agent_steps[1].argument_fingerprint == "a" * 64
    assert store.saved == [(response.conversation_id, "graph:trace-contract")]

    completed = next(
        record for record in caplog.records if record.getMessage() == "chat_completed"
    )
    assert completed.agent_step_count == 3
    assert completed.route_count == 3
    assert not hasattr(completed, "agent_steps")
    assert not hasattr(completed, "route_history")
    assert private_query not in " ".join(str(value) for value in completed.__dict__.values())
