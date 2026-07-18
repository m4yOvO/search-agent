"""Failure-tolerant exact-query cache backed by a Chroma server."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import Settings
from app.ids import stable_hash
from app.memory.canonicalizer import (
    canonical_query_id,
    deterministic_embedding,
    raw_query_hash,
)
from app.memory.graph_ops import evidence_coverage, graph_id_for, stable_evidence_rows
from app.schemas import (
    CacheScope,
    CacheStatus,
    CachedPayload,
    Evidence,
    GraphPayload,
    MemoryOperation,
    QuerySignature,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheLookup:
    hit: bool = False
    match_type: str | None = None
    record_id: str | None = None
    status: CacheStatus | None = None
    hit_count: int = 0
    payload: CachedPayload | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class CacheWriteResult:
    success: bool
    operation: MemoryOperation
    record_id: str | None = None
    status: CacheStatus | None = None
    reason: str | None = None


async def _call(method: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Call a synchronous Chroma API without blocking the event loop."""

    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _metadata_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


class LongTermMemory:
    """Chroma adapter implementing raw/canonical exact lookup and WARM/HOT reuse."""

    def __init__(
        self,
        settings: Settings,
        *,
        data_version: str,
        client: Any | None = None,
        collection: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.data_version = data_version
        self.client = client
        self.collection = collection
        self.collection_name = collection_name or settings.chroma_collection_name
        self.last_error: str | None = None

    async def initialize(self, *, required: bool = False) -> bool:
        """Connect with bounded retries; leave the application in fallback mode on failure."""

        if self.collection is not None:
            return True
        attempts = max(1, self.settings.chroma_connect_retries)
        for attempt in range(1, attempts + 1):
            try:
                if self.client is None:
                    import chromadb  # Imported lazily so pure unit tests stay lightweight.

                    self.client = await asyncio.to_thread(
                        chromadb.HttpClient,
                        host=self.settings.chroma_host,
                        port=self.settings.chroma_port,
                    )
                await _call(self.client.heartbeat)
                self.collection = await _call(
                    self.client.get_or_create_collection,
                    name=self.collection_name,
                )
                self.last_error = None
                return True
            except Exception as exc:  # Chroma exposes transport-specific exception types.
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Chroma connection attempt failed",
                    extra={"event": "cache_connect_failed", "attempt": attempt},
                )
                self.collection = None
                if attempt < attempts:
                    await asyncio.sleep(self.settings.chroma_retry_delay_seconds)
        if required:
            raise RuntimeError(f"unable to connect to Chroma: {self.last_error}")
        return False

    async def ping(self) -> bool:
        if self.client is None or self.collection is None:
            return False
        try:
            await _call(self.client.heartbeat)
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    async def close(self) -> None:
        if self.client is None:
            return
        close = getattr(self.client, "close", None)
        if close is not None:
            try:
                await _call(close)
            except Exception:
                logger.debug("Chroma client close failed", exc_info=True)
        self.collection = None
        self.client = None

    def raw_hash(self, query: str, locale: str) -> str:
        return raw_query_hash(query, locale, self.settings.permission_scope)

    def canonical_id(self, signature: QuerySignature) -> str:
        return canonical_query_id(
            signature,
            data_version=self.data_version,
            graph_schema_version=self.settings.graph_schema_version,
            permission_scope=self.settings.permission_scope,
        )

    async def lookup_raw(self, query: str, locale: str) -> CacheLookup:
        if self.collection is None:
            return CacheLookup(error="cache_unavailable")
        query_hash = self.raw_hash(query, locale)
        try:
            result = await _call(
                self.collection.get,
                where={"raw_query_hash": {"$eq": query_hash}},
                include=["documents", "metadatas"],
            )
            return await self._parse_result(
                result,
                match_type="raw_exact",
                expected_raw_hash=query_hash,
                expected_locale=locale,
            )
        except Exception as exc:
            return self._read_failure(exc, "raw_exact")

    async def lookup_canonical(self, signature: QuerySignature) -> CacheLookup:
        if self.collection is None:
            return CacheLookup(error="cache_unavailable")
        record_id = self.canonical_id(signature)
        try:
            result = await _call(
                self.collection.get,
                ids=[record_id],
                include=["documents", "metadatas"],
            )
            return await self._parse_result(
                result,
                match_type="canonical_exact",
                expected_canonical_id=record_id,
                expected_signature=signature,
            )
        except Exception as exc:
            return self._read_failure(exc, "canonical_exact")

    def _read_failure(self, exc: Exception, match_type: str) -> CacheLookup:
        self.last_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Chroma cache read failed; continuing through Researcher",
            extra={"event": "cache_read_failed", "match_type": match_type},
            exc_info=True,
        )
        return CacheLookup(error=self.last_error)

    async def _parse_result(
        self,
        result: Any,
        *,
        match_type: str,
        expected_raw_hash: str | None = None,
        expected_locale: str | None = None,
        expected_canonical_id: str | None = None,
        expected_signature: QuerySignature | None = None,
    ) -> CacheLookup:
        ids = list((result or {}).get("ids") or [])
        if not ids:
            return CacheLookup()
        documents = list((result or {}).get("documents") or [])
        metadatas = list((result or {}).get("metadatas") or [])
        last_failure = CacheLookup()
        # A raw-query hash can legitimately have rows from more than one dataset
        # version. Invalid/stale candidates must not shadow a later valid row.
        for index, raw_record_id in enumerate(ids):
            record_id = str(raw_record_id)
            metadata = (
                dict(metadatas[index] or {}) if index < len(metadatas) else {}
            )
            if (
                match_type == "raw_exact"
                and metadata.get("cache_scope") != CacheScope.CONTEXT_FREE.value
            ):
                # Raw aliases are globally reusable only for context-free queries.
                reason = "raw_scope_mismatch"
                await self._mark_stale(record_id, metadata, reason)
                last_failure = CacheLookup(record_id=record_id, error=reason)
                continue
            reason = self._invalid_metadata_reason(metadata)
            if reason:
                await self._mark_stale(record_id, metadata, reason)
                last_failure = CacheLookup(record_id=record_id, error=reason)
                continue
            try:
                document = documents[index] if index < len(documents) else ""
                payload = CachedPayload.model_validate(json.loads(document))
                self._validate_payload(
                    record_id,
                    metadata,
                    payload,
                    expected_raw_hash=expected_raw_hash,
                    expected_locale=expected_locale,
                    expected_canonical_id=expected_canonical_id,
                    expected_signature=expected_signature,
                )
            except Exception as exc:
                reason = f"invalid_cache_payload:{exc}"
                await self._mark_stale(record_id, metadata, reason)
                last_failure = CacheLookup(record_id=record_id, error=reason)
                continue
            status = CacheStatus(str(metadata["status"]))
            lookup = CacheLookup(
                hit=True,
                match_type=match_type,
                record_id=record_id,
                status=status,
                hit_count=int(metadata.get("hit_count", 0)),
                payload=payload,
                metadata=metadata,
            )
            logger.info(
                "cache hit",
                extra={
                    "event": "cache_hit",
                    "match_type": match_type,
                    "record_id": record_id,
                    "cache_status": status.value,
                },
            )
            return lookup
        return last_failure

    def _validate_payload(
        self,
        record_id: str,
        metadata: dict[str, Any],
        payload: CachedPayload,
        *,
        expected_raw_hash: str | None,
        expected_locale: str | None,
        expected_canonical_id: str | None,
        expected_signature: QuerySignature | None,
    ) -> None:
        signature = payload.query_signature
        if signature.version != self.settings.query_signature_version:
            raise ValueError("payload_signature_version_mismatch")
        if metadata.get("cache_scope") != payload.cache_scope.value:
            raise ValueError("payload_cache_scope_mismatch")
        if payload.graph.data_version != self.data_version:
            raise ValueError("payload_data_version_mismatch")
        if metadata.get("intent") != signature.intent.value:
            raise ValueError("payload_intent_mismatch")
        metadata_locale = str(metadata.get("locale", "")).casefold()
        if metadata_locale != signature.locale.casefold():
            raise ValueError("payload_locale_mismatch")
        if expected_locale is not None and metadata_locale != expected_locale.casefold():
            raise ValueError("raw_locale_mismatch")
        if expected_raw_hash is not None and metadata.get("raw_query_hash") != expected_raw_hash:
            raise ValueError("raw_query_hash_mismatch")

        canonical_id = self.canonical_id(signature)
        if record_id != canonical_id:
            raise ValueError("payload_canonical_id_mismatch")
        if expected_canonical_id is not None and record_id != expected_canonical_id:
            raise ValueError("requested_canonical_id_mismatch")
        if expected_signature is not None and signature != expected_signature:
            raise ValueError("requested_signature_mismatch")

        recomputed_graph_id = graph_id_for(
            payload.graph.nodes,
            payload.graph.edges,
            payload.graph.data_version,
            payload.graph.evidence,
        )
        if payload.graph.graph_id != recomputed_graph_id:
            raise ValueError("payload_graph_id_mismatch")
        if stable_evidence_rows(payload.evidence) != stable_evidence_rows(
            payload.graph.evidence
        ):
            raise ValueError("payload_evidence_catalog_mismatch")
        graph_node_ids = {node.id for node in payload.graph.nodes}
        if set(payload.focus_entity_ids) - graph_node_ids:
            raise ValueError("payload_focus_outside_graph")
        if evidence_coverage(payload.graph, payload.evidence) < 1.0:
            raise ValueError("payload_incomplete_evidence")
        if metadata.get("payload_hash") != self._payload_hash(payload):
            raise ValueError("payload_hash_mismatch")

    @staticmethod
    def _payload_hash(payload: CachedPayload) -> str:
        return stable_hash(payload.model_dump(mode="json"))

    def _invalid_metadata_reason(self, metadata: dict[str, Any]) -> str | None:
        now_epoch = int(datetime.now(UTC).timestamp())
        graph_schema_version = _metadata_int(metadata.get("graph_schema_version"))
        query_signature_version = _metadata_int(
            metadata.get("query_signature_version")
        )
        expires_at_epoch = _metadata_int(metadata.get("expires_at_epoch"))
        hit_count = _metadata_int(metadata.get("hit_count"))
        payload_hash = metadata.get("payload_hash")
        checks: tuple[tuple[bool, str], ...] = (
            (metadata.get("record_type") != "canonical_query_result", "invalid_record_type"),
            (metadata.get("status") not in {CacheStatus.WARM.value, CacheStatus.HOT.value}, "inactive_status"),
            (metadata.get("data_version") != self.data_version, "data_version_mismatch"),
            (
                graph_schema_version != self.settings.graph_schema_version,
                "graph_schema_version_mismatch",
            ),
            (
                query_signature_version != self.settings.query_signature_version,
                "query_signature_version_mismatch",
            ),
            (
                metadata.get("permission_scope") != self.settings.permission_scope,
                "permission_scope_mismatch",
            ),
            (
                metadata.get("cache_scope")
                not in {CacheScope.CONTEXT_FREE.value, CacheScope.CONVERSATION.value},
                "invalid_cache_scope",
            ),
            (hit_count is None or hit_count < 0, "invalid_hit_count"),
            (
                not isinstance(payload_hash, str)
                or len(payload_hash) != 64
                or any(character not in "0123456789abcdef" for character in payload_hash),
                "invalid_payload_hash",
            ),
            (
                expires_at_epoch is None or expires_at_epoch <= now_epoch,
                "cache_expired",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    async def _mark_stale(
        self, record_id: str, metadata: dict[str, Any], reason: str
    ) -> None:
        if self.collection is None:
            return
        updated = dict(metadata)
        updated["status"] = CacheStatus.STALE.value
        updated["stale_reason"] = reason[:500]
        try:
            await _call(self.collection.update, ids=[record_id], metadatas=[updated])
        except Exception:
            logger.warning(
                "Unable to mark invalid cache entry stale",
                extra={"event": "cache_stale_mark_failed", "record_id": record_id},
                exc_info=True,
            )

    async def touch(self, lookup: CacheLookup) -> CacheWriteResult:
        """Count exact reuse and promote WARM to HOT without extending its TTL."""

        if not lookup.hit or not lookup.record_id or not lookup.metadata:
            return CacheWriteResult(False, MemoryOperation.SKIP, reason="invalid_touch")
        if self.collection is None:
            return CacheWriteResult(False, MemoryOperation.SKIP, reason="cache_unavailable")
        metadata = dict(lookup.metadata)
        old_status = CacheStatus(str(metadata["status"]))
        new_status = CacheStatus.HOT
        metadata["status"] = new_status.value
        metadata["hit_count"] = int(metadata.get("hit_count", 0)) + 1
        metadata["last_accessed_at_epoch"] = int(datetime.now(UTC).timestamp())
        try:
            await _call(
                self.collection.update,
                ids=[lookup.record_id],
                metadatas=[metadata],
            )
            operation = (
                MemoryOperation.PROMOTE
                if old_status is CacheStatus.WARM
                else MemoryOperation.TOUCH
            )
            return CacheWriteResult(
                True,
                operation,
                record_id=lookup.record_id,
                status=new_status,
                reason="exact_cache_reuse",
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Chroma cache touch failed",
                extra={"event": "cache_touch_failed", "record_id": lookup.record_id},
                exc_info=True,
            )
            return CacheWriteResult(
                False, MemoryOperation.SKIP, record_id=lookup.record_id, reason=self.last_error
            )

    async def write(
        self,
        *,
        raw_query: str,
        locale: str,
        signature: QuerySignature,
        answer: str,
        graph: GraphPayload,
        evidence: list[Evidence],
        focus_entity_ids: list[str] | None = None,
        resolved_entities: dict[str, str] | None = None,
        cache_scope: CacheScope = CacheScope.CONTEXT_FREE,
    ) -> CacheWriteResult:
        if self.collection is None:
            return CacheWriteResult(False, MemoryOperation.SKIP, reason="memory_write_failed:cache_unavailable")
        if signature.version != self.settings.query_signature_version:
            return CacheWriteResult(
                False, MemoryOperation.SKIP, reason="invalid_query_signature_version"
            )
        if signature.locale.casefold() != locale.casefold():
            return CacheWriteResult(
                False, MemoryOperation.SKIP, reason="query_signature_locale_mismatch"
            )
        if graph.data_version != self.data_version:
            return CacheWriteResult(
                False, MemoryOperation.SKIP, reason="graph_data_version_mismatch"
            )
        record_id = self.canonical_id(signature)
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.settings.cache_ttl_hours)
        payload = CachedPayload(
            answer=answer,
            graph=graph,
            evidence=evidence,
            query_signature=signature,
            focus_entity_ids=focus_entity_ids or [],
            resolved_entities=resolved_entities or {},
            cache_scope=cache_scope,
        )
        metadata: dict[str, str | int | float | bool] = {
            "record_type": "canonical_query_result",
            "intent": signature.intent.value,
            "locale": locale,
            "cache_scope": cache_scope.value,
            "status": CacheStatus.WARM.value,
            "hit_count": 0,
            "data_version": self.data_version,
            "graph_schema_version": self.settings.graph_schema_version,
            "query_signature_version": self.settings.query_signature_version,
            "permission_scope": self.settings.permission_scope,
            "created_at_epoch": int(now.timestamp()),
            "last_accessed_at_epoch": int(now.timestamp()),
            "expires_at_epoch": int(expires_at.timestamp()),
            "payload_hash": self._payload_hash(payload),
        }
        if cache_scope is CacheScope.CONTEXT_FREE:
            metadata["raw_query_hash"] = self.raw_hash(raw_query, locale)
        try:
            self._validate_payload(
                record_id,
                metadata,
                payload,
                expected_raw_hash=(
                    str(metadata["raw_query_hash"])
                    if cache_scope is CacheScope.CONTEXT_FREE
                    else None
                ),
                expected_locale=locale,
                expected_canonical_id=record_id,
                expected_signature=signature,
            )
        except ValueError as exc:
            return CacheWriteResult(
                False,
                MemoryOperation.SKIP,
                record_id=record_id,
                reason=f"invalid_cache_write:{exc}",
            )
        try:
            await _call(
                self.collection.upsert,
                ids=[record_id],
                embeddings=[deterministic_embedding(signature)],
                documents=[json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)],
                metadatas=[metadata],
            )
            logger.info(
                "verified query result cached",
                extra={"event": "cache_write", "record_id": record_id, "cache_status": "warm"},
            )
            return CacheWriteResult(
                True,
                MemoryOperation.ADD,
                record_id=record_id,
                status=CacheStatus.WARM,
                reason="first_verified_result",
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Chroma cache write failed; returning current answer",
                extra={"event": "memory_write_failed", "record_id": record_id},
                exc_info=True,
            )
            return CacheWriteResult(
                False,
                MemoryOperation.SKIP,
                record_id=record_id,
                reason=f"memory_write_failed:{self.last_error}",
            )
