from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import open_application_service
from tests.scenarios import build_acceptance_model


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_CHROMA_INTEGRATION") != "1",
    reason="set RUN_CHROMA_INTEGRATION=1 with a real Chroma server",
)
def test_real_chroma_three_turn_acceptance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mandatory cache flow against Chroma, with a run-unique namespace."""

    data_directory = Path(__file__).resolve().parents[3] / "data"
    settings = Settings(
        data_directory=data_directory,
        runtime_directory=tmp_path,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        graph_store_path=tmp_path / "graphs.sqlite3",
        chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
        chroma_collection_prefix=f"integration_{uuid4().hex}",
        chroma_connect_retries=2,
        chroma_retry_delay_seconds=0.05,
    )
    model = asyncio.run(
        build_acceptance_model(
            data_directory,
            include_followup=True,
            include_related_query=True,
        )
    )

    @asynccontextmanager
    async def service_factory():
        async with open_application_service(settings, model_client=model) as service:
            yield service

    with TestClient(
        create_app(settings=settings, service_factory=service_factory)
    ) as client:
        assert client.get("/ready").status_code == 200

        first = client.post("/chat", json={"message": "马云创办了哪些公司？"})
        assert first.status_code == 200, first.text
        first_body = first.json()
        conversation_id = first_body["conversation_id"]
        first_labels = {node["label"] for node in first_body["graph"]["nodes"]}
        assert "马云" in first_labels
        assert "阿里巴巴集团" in first_labels
        founded_edges = [
            edge for edge in first_body["graph"]["edges"] if edge["type"] == "founded"
        ]
        assert founded_edges
        assert {edge["properties"]["raw_relation"] for edge in founded_edges} == {
            "Founder_of"
        }
        assert not [
            edge for edge in first_body["graph"]["edges"] if edge["type"] == "controls"
        ]
        assert first_body["memory"]["cache_hit"] is False
        assert first_body["memory"]["status"] == "warm"
        assert first_body["memory"]["write_operation"] == "add"
        assert first_body["trace"]["researcher_invoked"] is True
        assert first_body["trace"]["model_calls"] == 5

        second = client.post(
            "/chat",
            json={
                "conversation_id": conversation_id,
                "message": "这些公司在哪？",
            },
        )
        assert second.status_code == 200, second.text
        second_body = second.json()
        assert any(
            node["type"] == "location" for node in second_body["graph"]["nodes"]
        )
        assert any(
            edge["type"] == "headquartered_in"
            for edge in second_body["graph"]["edges"]
        )

        third = client.post(
            "/chat",
            json={
                "conversation_id": conversation_id,
                "message": "马云创办了哪些公司？",
            },
        )
        assert third.status_code == 200, third.text
        third_body = third.json()
        assert third_body["memory"]["cache_hit"] is True
        assert third_body["memory"]["match_type"] == "raw_exact"
        assert third_body["memory"]["status"] == "hot"
        assert third_body["trace"]["researcher_invoked"] is False
        assert third_body["trace"]["tool_calls"] == 0
        assert third_body["trace"]["model_calls"] == 0
        assert client.get(
            "/graph", params={"conversation_id": conversation_id}
        ).status_code == 200

        related = client.post(
            "/chat", json={"message": "阿里巴巴集团拥有哪些公司？"}
        )
        assert related.status_code == 200, related.text
        related_nodes = related.json()["graph"]["nodes"]
        assert {node["label"] for node in related_nodes} >= {
            "阿里巴巴集团",
            "阿里云",
        }
        assert any(edge["type"] == "owns" for edge in related.json()["graph"]["edges"])

    captured_logs = capsys.readouterr().out
    assert '"event": "cache_hit"' in captured_logs
