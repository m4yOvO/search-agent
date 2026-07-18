from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.graph_store import GraphSnapshotStore
from app.main import create_app
from app.service import ApplicationService, ServiceResources


class HealthyComponent:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    async def ping(self) -> bool:
        return self.healthy


class FakeCompiledGraph:
    """StateGraph test double; API tests assert its single ainvoke boundary."""

    def __init__(self) -> None:
        self.invocations: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.query_counts: defaultdict[str, int] = defaultdict(int)
        self.session_nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.session_edges: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.invalid_output = False
        self.raise_error = False
        self.terminal_failure = False

    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.invocations.append((dict(input), dict(config)))
        if self.raise_error:
            raise RuntimeError("internal detail must not leak")
        if self.invalid_output:
            return {"answer": "missing graph"}

        query = input["current_query"]
        conversation_id = input["conversation_id"]
        self.query_counts[query] += 1
        is_hit = self.query_counts[query] > 1 and query == "马云创办了哪些公司？"

        nodes = self.session_nodes[conversation_id]
        edges = self.session_edges[conversation_id]
        nodes.update(
            {
                "person:P004": {
                    "id": "person:P004",
                    "type": "person",
                    "label": "马云",
                    "properties": {"source_id": "P004", "demo": True},
                    "evidence_ids": ["evidence:raw:person:P004"],
                },
                "company:C005": {
                    "id": "company:C005",
                    "type": "company",
                    "label": "阿里巴巴集团",
                    "properties": {"source_id": "C005", "demo": True},
                    "evidence_ids": ["evidence:raw:company:C005"],
                },
            }
        )
        edges.update(
            {
                "relation:raw:0006": {
                    "id": "relation:raw:0006",
                    "source": "person:P004",
                    "target": "company:C005",
                    "type": "founded",
                    "label": "Founder_of",
                    "properties": {"raw_relation": "Founder_of", "demo": True},
                    "evidence_ids": ["evidence:raw:relation:0006"],
                },
            }
        )

        if query == "这些公司在哪？":
            nodes.update(
                {
                    "location:hangzhou": {
                        "id": "location:hangzhou",
                        "type": "location",
                        "label": "Hangzhou",
                        "properties": {"demo": True},
                        "evidence_ids": ["evidence:raw:location:hangzhou"],
                    },
                }
            )
            edges.update(
                {
                    "relation:raw:0064": {
                        "id": "relation:raw:0064",
                        "source": "company:C005",
                        "target": "location:hangzhou",
                        "type": "headquartered_in",
                        "label": "Headquartered_in",
                        "properties": {"raw_relation": "Headquartered_in", "demo": True},
                        "evidence_ids": ["evidence:raw:relation:0064"],
                    },
                }
            )

        referenced_evidence_ids = sorted(
            {
                evidence_id
                for element in [*nodes.values(), *edges.values()]
                for evidence_id in element["evidence_ids"]
            }
        )
        evidence = [
            {
                "id": evidence_id,
                "provider": "api-test-fixture",
                "record_id": evidence_id.removeprefix("evidence:"),
                "source_kind": "demo_fixture",
                "updated_at": "2026-07-17T00:00:00Z",
                "retrieved_at": "2026-07-17T00:00:00Z",
                "is_demo": True,
                "source_url": None,
            }
            for evidence_id in referenced_evidence_ids
        ]

        result = {
            "answer": (
                "阿里巴巴集团位于 Hangzhou（本地演示数据）。"
                if query == "这些公司在哪？"
                else "原始演示数据中，马云创办了阿里巴巴集团。"
            ),
            "session_graph": {
                # Graph IDs identify graph content, not a conversation.  Two sessions
                # with the same facts intentionally exercise the store's many-to-one
                # conversation-head mapping.
                "graph_id": (
                    "graph:companies-and-locations"
                    if any(node_id.startswith("location:") for node_id in nodes)
                    else "graph:founded-company"
                ),
                "nodes": list(nodes.values()),
                "edges": list(edges.values()),
                "evidence": evidence,
                "data_version": "raw-v1",
            },
            "cache_metadata": {
                "cache_hit": is_hit,
                "tier": "long_term" if is_hit else None,
                "match_type": "raw_exact" if is_hit else None,
                "status": "hot" if is_hit else "warm",
                "write_operation": "promote" if is_hit else "add",
                "result_id": "cache:test",
            },
            "researcher_invoked": not is_hit,
            "tool_call_count": 0 if is_hit else 2,
            "research_steps": 0 if is_hit else 2,
            "replan_count": 0,
            "route_history": (
                ["begin_turn", "raw_cache_probe", "cache_hydrate", "visualizer"]
                if is_hit
                else ["begin_turn", "raw_cache_probe", "planner", "researcher", "visualizer"]
            ),
        }
        if self.terminal_failure:
            result.update(
                {
                    "answer": "本次查询未能从本地演示工具生成可验证结果。",
                    "run_status": "failed",
                    "llm_errors": ["sanitized model failure"],
                }
            )
        return result


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        runtime_directory=tmp_path,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        graph_store_path=tmp_path / "graphs.sqlite3",
        chroma_collection_prefix="test_query_cache",
    )


@pytest.fixture
def fake_graph() -> FakeCompiledGraph:
    return FakeCompiledGraph()


@pytest.fixture
def app_client(
    settings: Settings, fake_graph: FakeCompiledGraph
) -> AsyncIterator[TestClient]:
    @asynccontextmanager
    async def service_factory() -> AsyncIterator[ApplicationService]:
        store = await GraphSnapshotStore.open(settings.graph_store_path)
        try:
            yield ApplicationService(
                ServiceResources(
                    graph=fake_graph,
                    graph_store=store,
                    settings=settings,
                    cache=HealthyComponent(),
                    model=SimpleNamespace(provider="openai", model_name="test-model"),
                    checkpointer=object(),
                )
            )
        finally:
            await store.close()

    application = create_app(settings=settings, service_factory=service_factory)
    with TestClient(application) as client:
        yield client
