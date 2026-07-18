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


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_real_runtime_falls_back_when_chroma_is_unavailable(tmp_path: Path) -> None:
    data_directory = Path(__file__).resolve().parents[3] / "data"
    settings = Settings(
        data_directory=data_directory,
        runtime_directory=tmp_path,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        graph_store_path=tmp_path / "graphs.sqlite3",
        chroma_host="127.0.0.1",
        chroma_port=_unused_local_port(),
        chroma_collection_prefix="fallback_test",
        chroma_connect_retries=1,
        chroma_retry_delay_seconds=0,
    )
    model = asyncio.run(
        build_acceptance_model(data_directory, include_followup=False)
    )

    @asynccontextmanager
    async def service_factory():
        async with open_application_service(settings, model_client=model) as service:
            yield service

    with TestClient(
        create_app(settings=settings, service_factory=service_factory)
    ) as client:
        readiness = client.get("/ready")
        response = client.post("/chat", json={"message": "马云创办了哪些公司？"})

    assert readiness.status_code == 503
    assert readiness.json()["checks"]["chroma"] is False
    assert response.status_code == 200, response.text
    body = response.json()
    labels = {node["label"] for node in body["graph"]["nodes"]}
    assert {"马云", "阿里巴巴集团"} <= labels
    assert any(edge["type"] == "founded" for edge in body["graph"]["edges"])
    assert body["memory"]["cache_hit"] is False
    assert body["memory"]["write_operation"] == "skip"
    assert body["memory"]["reason"].startswith("memory_write_failed")
    assert body["trace"]["researcher_invoked"] is True
