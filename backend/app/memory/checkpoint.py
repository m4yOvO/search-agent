"""SQLite checkpointer lifecycle helper for the FastAPI lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any

import aiosqlite

from app.memory.chroma_store import CacheLookup
from app.schemas import (
    CacheMetadata,
    CacheScope,
    CacheStatus,
    CachedPayload,
    ConversationSummary,
    ConversationTurn,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphPayload,
    Intent,
    MemoryOperation,
    NodeType,
    PlannerDecision,
    QuerySignature,
    RelationType,
    ResearchAction,
    ResearcherDecision,
    ResearchTask,
    ToolError,
    ToolName,
    ToolResult,
    TraceMetadata,
    VisualizerDecision,
)


# LangGraph 1.2 warns that permissive import-based checkpoint revival will be
# disabled in a future release. Keep the SQLite boundary explicit: only the
# application-owned value types that can appear in current session state or in
# checkpoints created before request channels became untracked may be rebuilt.
CHECKPOINT_ALLOWED_TYPES: tuple[type[Any], ...] = (
    CacheLookup,
    CacheMetadata,
    CacheScope,
    CacheStatus,
    CachedPayload,
    ConversationSummary,
    ConversationTurn,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphPayload,
    Intent,
    MemoryOperation,
    NodeType,
    PlannerDecision,
    QuerySignature,
    RelationType,
    ResearchAction,
    ResearcherDecision,
    ResearchTask,
    ToolError,
    ToolName,
    ToolResult,
    TraceMetadata,
    VisualizerDecision,
)


@asynccontextmanager
async def open_checkpointer(path: str | Path) -> AsyncIterator[Any]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = tuple(
        (value_type.__module__, value_type.__name__)
        for value_type in CHECKPOINT_ALLOWED_TYPES
    )
    serializer = JsonPlusSerializer(
        allowed_json_modules=symbols,
        allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES,
    )
    async with aiosqlite.connect(str(checkpoint_path)) as connection:
        saver = AsyncSqliteSaver(connection, serde=serializer)
        await saver.setup()
        yield saver
