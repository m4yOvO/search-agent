from __future__ import annotations

import socket
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import open_application_service
from tests.scenarios import build_acceptance_model


pytestmark = pytest.mark.integration


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_checkpoint_and_graph_head_survive_application_restart(tmp_path: Path) -> None:
    """Models a backend named-volume restart without relying on long-term cache."""

    data_directory = Path(__file__).resolve().parents[3] / "data"
    settings = Settings(
        data_directory=data_directory,
        runtime_directory=tmp_path,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        graph_store_path=tmp_path / "graphs.sqlite3",
        chroma_host="127.0.0.1",
        chroma_port=_closed_port(),
        chroma_collection_prefix="restart_fallback_test",
        chroma_connect_retries=1,
        chroma_retry_delay_seconds=0,
    )
    model = asyncio.run(build_acceptance_model(data_directory))

    @asynccontextmanager
    async def service_factory():
        async with open_application_service(settings, model_client=model) as service:
            yield service

    with TestClient(
        create_app(settings=settings, service_factory=service_factory)
    ) as first_client:
        first = first_client.post(
            "/chat", json={"message": "马云创办了哪些公司？"}
        )
        assert first.status_code == 200, first.text
        conversation_id = first.json()["conversation_id"]
        first_graph_id = first.json()["graph_id"]

    # A new lifespan owns new SQLite connections but uses the same persisted files.
    with TestClient(
        create_app(settings=settings, service_factory=service_factory)
    ) as second_client:
        restored = second_client.get(
            "/graph", params={"conversation_id": conversation_id}
        )
        assert restored.status_code == 200
        assert restored.json()["graph_id"] == first_graph_id

        follow_up = second_client.post(
            "/chat",
            json={
                "conversation_id": conversation_id,
                "message": "这些公司在哪？",
            },
        )
        assert follow_up.status_code == 200, follow_up.text
        body = follow_up.json()
        assert any(node["type"] == "location" for node in body["graph"]["nodes"])
        assert any(
            edge["type"] == "headquartered_in" for edge in body["graph"]["edges"]
        )
