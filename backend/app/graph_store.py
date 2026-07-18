"""Durable SQLite storage for conversation graph snapshots.

LangGraph's checkpointer owns conversational state.  This deliberately separate store
provides the public ``/graph`` lookup contract without exposing checkpoint internals.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from .ids import canonical_json
from .memory.graph_ops import stable_evidence_rows
from .schemas import GraphPayload


class GraphSnapshotStore:
    """A single-connection asynchronous SQLite graph snapshot repository."""

    def __init__(self, path: Path, connection: aiosqlite.Connection) -> None:
        self.path = path
        self._connection = connection
        self._write_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, path: str | Path) -> GraphSnapshotStore:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(resolved)
        connection.row_factory = aiosqlite.Row
        store = cls(resolved, connection)
        try:
            await store._initialize()
        except BaseException:
            await connection.close()
            raise
        return store

    async def _initialize(self) -> None:
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA busy_timeout=5000")
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_payloads (
                graph_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                saved_at TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_graph_heads (
                conversation_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                saved_at TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (graph_id) REFERENCES graph_payloads (graph_id)
            )
            """
        )
        await self._connection.commit()

    async def save(self, conversation_id: str, graph: GraphPayload) -> None:
        """Upsert a de-duplicated graph and independently advance one session head."""

        if self._closed:
            raise RuntimeError("graph snapshot store is closed")
        if not conversation_id:
            raise ValueError("conversation_id must not be empty")

        # Round-trip through the strict model before any persistent write.  This also
        # guarantees that invalid edge endpoints never reach the public graph store.
        validated = GraphPayload.model_validate(graph)
        payload_json = validated.model_dump_json()
        async with self._write_lock:
            try:
                cursor = await self._connection.execute(
                    "SELECT payload_json FROM graph_payloads WHERE graph_id = ?",
                    (validated.graph_id,),
                )
                try:
                    existing_row = await cursor.fetchone()
                finally:
                    await cursor.close()
                if existing_row:
                    existing = GraphPayload.model_validate_json(
                        existing_row["payload_json"]
                    )
                    if self._content_signature(existing) != self._content_signature(
                        validated
                    ):
                        raise ValueError(
                            "graph_id collision: the same ID has different graph content"
                        )

                await self._connection.execute(
                    """
                    INSERT INTO graph_payloads (graph_id, payload_json, saved_at)
                    VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    ON CONFLICT(graph_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        saved_at = excluded.saved_at
                    """,
                    (validated.graph_id, payload_json),
                )
                await self._connection.execute(
                    """
                    INSERT INTO conversation_graph_heads (conversation_id, graph_id, saved_at)
                    VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        graph_id = excluded.graph_id,
                        saved_at = excluded.saved_at
                    """,
                    (conversation_id, validated.graph_id),
                )
                await self._connection.commit()
            except BaseException:
                await self._connection.rollback()
                raise

    @staticmethod
    def _content_signature(graph: GraphPayload) -> tuple[object, ...]:
        """Compare stable graph identity, ignoring fetch and generation times."""

        nodes = tuple(
            sorted(
                (
                    canonical_json(node.model_dump(mode="json"))
                    for node in graph.nodes
                )
            )
        )
        edges = tuple(
            sorted(
                (
                    canonical_json(edge.model_dump(mode="json"))
                    for edge in graph.edges
                )
            )
        )
        evidence = tuple(
            canonical_json(item) for item in stable_evidence_rows(graph.evidence)
        )
        return graph.data_version, nodes, edges, evidence

    async def get_by_graph_id(self, graph_id: str) -> GraphPayload | None:
        if self._closed:
            raise RuntimeError("graph snapshot store is closed")
        cursor = await self._connection.execute(
            "SELECT payload_json FROM graph_payloads WHERE graph_id = ?",
            (graph_id,),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        return GraphPayload.model_validate_json(row["payload_json"]) if row else None

    async def get_latest_for_conversation(
        self, conversation_id: str
    ) -> GraphPayload | None:
        if self._closed:
            raise RuntimeError("graph snapshot store is closed")
        cursor = await self._connection.execute(
            """
            SELECT payload.payload_json
            FROM conversation_graph_heads AS head
            JOIN graph_payloads AS payload ON payload.graph_id = head.graph_id
            WHERE head.conversation_id = ?
            """,
            (conversation_id,),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        return GraphPayload.model_validate_json(row["payload_json"]) if row else None

    async def ping(self) -> bool:
        if self._closed:
            return False
        try:
            cursor = await self._connection.execute("SELECT 1")
            try:
                row = await cursor.fetchone()
            finally:
                await cursor.close()
            return row is not None and row[0] == 1
        except (aiosqlite.Error, RuntimeError):
            return False

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._connection.close()

    async def __aenter__(self) -> GraphSnapshotStore:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
