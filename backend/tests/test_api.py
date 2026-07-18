from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.graph_store import GraphSnapshotStore
from app.main import create_app
from app.service import ApplicationService, ServiceResources

from .conftest import FakeCompiledGraph, HealthyComponent


QUERY = "马云创办了哪些公司？"


def _evidence(evidence_id: str, *, provider: str = "api-test-fixture") -> dict[str, object]:
    return {
        "id": evidence_id,
        "provider": provider,
        "record_id": evidence_id.removeprefix("evidence:"),
        "source_kind": "demo_fixture",
        "updated_at": "2026-07-17T00:00:00Z",
        "retrieved_at": "2026-07-17T00:00:00Z",
        "is_demo": True,
        "source_url": None,
    }


def test_health_and_ready(app_client: TestClient) -> None:
    health = app_client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "enterprise-relationship-explorer",
    }

    ready = app_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert all(ready.json()["checks"].values())


def test_cors_only_allows_configured_development_origin(app_client: TestClient) -> None:
    allowed = app_client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = app_client.options(
        "/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers


def test_chat_calls_compiled_graph_once_and_persists_graph(
    app_client: TestClient, fake_graph: FakeCompiledGraph, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    response = app_client.post("/chat", json={"message": QUERY, "locale": "zh-CN"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["error_code"] is None
    assert body["disclaimer"].startswith("结果来自本地演示数据")
    assert {node["label"] for node in body["graph"]["nodes"]} >= {
        "马云",
        "阿里巴巴集团",
    }
    assert body["memory"]["cache_hit"] is False
    assert body["trace"]["researcher_invoked"] is True
    evidence_ids = {item["id"] for item in body["graph"]["evidence"]}
    assert evidence_ids
    assert all(
        set(element["evidence_ids"]) <= evidence_ids
        for element in [*body["graph"]["nodes"], *body["graph"]["edges"]]
    )
    assert len(fake_graph.invocations) == 1
    graph_input, config = fake_graph.invocations[0]
    assert graph_input["conversation_id"] == body["conversation_id"]
    assert graph_input["current_query"] == QUERY
    assert config["configurable"]["thread_id"] == body["conversation_id"]
    assert config["recursion_limit"] > 0
    assert "cache_miss" in caplog.messages

    by_id = app_client.get("/graph", params={"graph_id": body["graph_id"]})
    by_conversation = app_client.get(
        "/graph", params={"conversation_id": body["conversation_id"]}
    )
    assert by_id.status_code == 200
    assert by_conversation.status_code == 200
    assert by_id.json() == body["graph"]
    assert by_conversation.json() == body["graph"]


def test_three_turn_response_shape_and_cache_log(
    app_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    conversation_id = str(uuid4())

    first = app_client.post(
        "/chat", json={"conversation_id": conversation_id, "message": QUERY}
    )
    second = app_client.post(
        "/chat",
        json={"conversation_id": conversation_id, "message": "这些公司在哪？"},
    )
    third = app_client.post(
        "/chat", json={"conversation_id": conversation_id, "message": QUERY}
    )

    assert first.status_code == second.status_code == third.status_code == 200
    assert first.json()["memory"]["status"] == "warm"
    second_body = second.json()
    assert {node["type"] for node in second_body["graph"]["nodes"]} >= {
        "person",
        "company",
        "location",
    }
    assert any(
        edge["type"] == "headquartered_in"
        for edge in second_body["graph"]["edges"]
    )
    third_body = third.json()
    assert third_body["memory"]["cache_hit"] is True
    assert third_body["memory"]["match_type"] == "raw_exact"
    assert third_body["memory"]["status"] == "hot"
    assert third_body["trace"]["researcher_invoked"] is False
    assert third_body["trace"]["tool_calls"] == 0
    assert "cache_hit" in caplog.messages


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"message": ""}, 422),
        ({"message": "   "}, 422),
        ({"message": QUERY, "unexpected": True}, 422),
        ({"message": QUERY, "conversation_id": "not-a-uuid"}, 422),
        ({"message": "x" * 1001}, 422),
    ],
)
def test_chat_validation_errors(
    app_client: TestClient, payload: dict[str, object], expected_status: int
) -> None:
    assert app_client.post("/chat", json=payload).status_code == expected_status


def test_graph_selector_errors_and_unknowns(app_client: TestClient) -> None:
    assert app_client.get("/graph").status_code == 422
    assert (
        app_client.get(
            "/graph",
            params={"graph_id": "graph:none", "conversation_id": str(uuid4())},
        ).status_code
        == 422
    )
    assert (
        app_client.get("/graph", params={"conversation_id": "not-a-uuid"}).status_code
        == 422
    )
    assert app_client.get("/graph", params={"graph_id": "graph:none"}).status_code == 404
    assert (
        app_client.get("/graph", params={"conversation_id": str(uuid4())}).status_code
        == 404
    )


def test_conversations_are_isolated(app_client: TestClient) -> None:
    first_id, second_id = str(uuid4()), str(uuid4())
    first = app_client.post(
        "/chat", json={"conversation_id": first_id, "message": QUERY}
    ).json()
    second = app_client.post(
        "/chat", json={"conversation_id": second_id, "message": QUERY}
    ).json()

    assert first["graph_id"] == second["graph_id"]
    assert (
        app_client.get("/graph", params={"conversation_id": first_id}).json()[
            "graph_id"
        ]
        == first["graph_id"]
    )
    assert (
        app_client.get("/graph", params={"conversation_id": second_id}).json()[
            "graph_id"
        ]
        == second["graph_id"]
    )


def test_graph_execution_failure_is_sanitized(
    app_client: TestClient, fake_graph: FakeCompiledGraph
) -> None:
    fake_graph.raise_error = True
    response = app_client.post("/chat", json={"message": QUERY})
    assert response.status_code == 500
    assert response.json()["detail"] == "state graph execution failed"
    assert "internal detail" not in response.text


def test_terminal_agent_failure_is_typed_and_safe(
    app_client: TestClient, fake_graph: FakeCompiledGraph
) -> None:
    fake_graph.terminal_failure = True
    response = app_client.post("/chat", json={"message": QUERY})

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_code"] == "model_failure"
    assert "sanitized model failure" not in response.text


def test_complete_agent_turn_timeout_returns_504(
    settings: Settings, tmp_path: Path
) -> None:
    class SlowCompiledGraph:
        async def ainvoke(self, input, config):
            await asyncio.sleep(0.05)
            return {}

    timeout_settings = settings.model_copy(update={"chat_timeout_seconds": 0.001})

    @asynccontextmanager
    async def service_factory():
        store = await GraphSnapshotStore.open(tmp_path / "timeout-graphs.sqlite3")
        try:
            yield ApplicationService(
                ServiceResources(
                    graph=SlowCompiledGraph(),
                    graph_store=store,
                    settings=timeout_settings,
                )
            )
        finally:
            await store.close()

    app = create_app(settings=timeout_settings, service_factory=service_factory)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": QUERY})

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "state graph execution exceeded the configured turn timeout"
    )


def test_invalid_state_graph_output_returns_500(
    app_client: TestClient, fake_graph: FakeCompiledGraph
) -> None:
    fake_graph.invalid_output = True
    response = app_client.post("/chat", json={"message": QUERY})
    assert response.status_code == 500
    assert response.json()["detail"] == "state graph returned an invalid public result"


def test_chroma_not_ready_does_not_block_chat(
    settings: Settings, tmp_path: Path
) -> None:
    graph = FakeCompiledGraph()

    @asynccontextmanager
    async def service_factory():
        store = await GraphSnapshotStore.open(tmp_path / "unready-graphs.sqlite3")
        try:
            yield ApplicationService(
                ServiceResources(
                    graph=graph,
                    graph_store=store,
                    settings=settings,
                    cache=HealthyComponent(False),
                    checkpointer=object(),
                )
            )
        finally:
            await store.close()

    with TestClient(create_app(settings=settings, service_factory=service_factory)) as client:
        ready = client.get("/ready")
        chat = client.post("/chat", json={"message": QUERY})

    assert ready.status_code == 503
    assert ready.json()["checks"]["chroma"] is False
    assert chat.status_code == 200


@pytest.mark.asyncio
async def test_graph_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persistent.sqlite3"
    graph = {
        "graph_id": "graph:persisted",
        "nodes": [
            {
                "id": "company:tesla",
                "type": "company",
                "label": "Tesla",
                "evidence_ids": ["evidence:company:tesla"],
            }
        ],
        "edges": [],
        "evidence": [_evidence("evidence:company:tesla")],
        "data_version": "demo-v1",
    }
    conversation_id = str(uuid4())
    first_store = await GraphSnapshotStore.open(path)
    from app.schemas import GraphPayload

    await first_store.save(conversation_id, GraphPayload.model_validate(graph))
    await first_store.close()

    second_store = await GraphSnapshotStore.open(path)
    restored = await second_store.get_latest_for_conversation(conversation_id)
    await second_store.close()
    assert restored is not None
    assert restored.graph_id == "graph:persisted"


@pytest.mark.asyncio
async def test_graph_store_rejects_same_id_with_different_content(tmp_path: Path) -> None:
    from app.schemas import GraphPayload

    store = await GraphSnapshotStore.open(tmp_path / "collision.sqlite3")
    base = {
        "graph_id": "graph:collision",
        "nodes": [
            {
                "id": "company:tesla",
                "type": "company",
                "label": "Tesla",
                "evidence_ids": ["evidence:company:tesla"],
            }
        ],
        "edges": [],
        "evidence": [_evidence("evidence:company:tesla")],
        "data_version": "demo-v1",
    }
    await store.save(str(uuid4()), GraphPayload.model_validate(base))
    changed = {**base, "nodes": [{**base["nodes"][0], "label": "Not Tesla"}]}
    with pytest.raises(ValueError, match="graph_id collision"):
        await store.save(str(uuid4()), GraphPayload.model_validate(changed))

    refreshed_evidence = {
        **base,
        "evidence": [
            {
                **_evidence("evidence:company:tesla"),
                "retrieved_at": "2026-07-18T00:00:00Z",
            }
        ],
    }
    await store.save(str(uuid4()), GraphPayload.model_validate(refreshed_evidence))

    changed_evidence = {
        **base,
        "evidence": [
            _evidence("evidence:company:tesla", provider="different-provider")
        ],
    }
    with pytest.raises(ValueError, match="graph_id collision"):
        await store.save(
            str(uuid4()), GraphPayload.model_validate(changed_evidence)
        )
    await store.close()
