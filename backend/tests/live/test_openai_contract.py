from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agents import AgentDependencies, compile_agent_graph
from app.config import Settings
from app.llm import OpenAIModelClient
from app.schemas import Intent, RelationType
from app.state_views import request_semantics
from app.tools import FixtureRepository, ToolRegistry


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_LIVE") != "1",
        reason="set RUN_OPENAI_LIVE=1 to run the paid provider contract test",
    ),
]


@pytest.mark.asyncio
async def test_openai_agents_use_mock_tools_and_emit_verified_graph() -> None:
    settings = Settings()
    model = OpenAIModelClient(
        api_key=settings.require_openai_api_key(),
        model_name=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    data_directory = Path(__file__).resolve().parents[3] / "data"
    repository = FixtureRepository.load(data_directory)
    tools = ToolRegistry(repository)
    graph = compile_agent_graph(
        AgentDependencies(
            settings=settings,
            tools=tools,
            cache=None,
            data_version=repository.data_version,
            model=model,
            planner_catalog=repository.compact_planner_catalog(),
            planner_tools=tuple(
                {"name": spec.name.value, "description": spec.description}
                for spec in tools.specs
            ),
        )
    )

    result = await graph.ainvoke(
        {
            "conversation_id": "live-openai-contract",
            "request_id": "live-request",
            "current_query": "马云创办了哪些公司？",
            "locale": "zh-CN",
        },
        config={"recursion_limit": settings.graph_recursion_limit},
    )

    labels = {node.label for node in result["session_graph"].nodes}
    founded_edges = [
        edge for edge in result["session_graph"].edges if edge.type.value == "founded"
    ]
    assert {"马云", "阿里巴巴集团"} <= labels
    assert founded_edges
    assert {edge.source for edge in founded_edges} == {"person:P004"}
    assert {edge.target for edge in founded_edges} == {"company:C005"}
    assert {
        edge.properties["raw_relation"] for edge in founded_edges
    } == {"Founder_of"}
    assert not [
        edge for edge in result["session_graph"].edges if edge.type.value == "controls"
    ]
    assert request_semantics(result).intent is Intent.FIND_RELATED_COMPANIES
    assert result["query_signature"].relation_types == [RelationType.FOUNDED]
    assert result["model_provider"] == "openai"
    assert result["model_call_count"] >= 3
    assert result["planner_model_calls"] == 1
    assert result["researcher_model_calls"] >= 1
    assert result["visualizer_model_calls"] == 1
    assert result["tool_call_count"] >= 2
    assert all(
        element.evidence_ids
        for element in [
            *result["session_graph"].nodes,
            *result["session_graph"].edges,
        ]
    )
