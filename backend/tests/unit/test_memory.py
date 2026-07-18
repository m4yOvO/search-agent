from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.memory.canonicalizer import (
    canonical_query_id,
    deterministic_embedding,
    raw_query_hash,
)
from app.memory.chroma_store import LongTermMemory
from app.memory.graph_ops import make_graph, merge_graphs
from app.memory.policy import decide_memory_write
from app.schemas import (
    CacheStatus,
    CacheScope,
    CachedPayload,
    ControlQueryPolicy,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphPayload,
    Intent,
    MemoryOperation,
    NodeType,
    QuerySignature,
    RelationType,
)


class FakeCollection:
    def __init__(self, *, fail_reads: bool = False, fail_writes: bool = False) -> None:
        self.rows: dict[str, dict] = {}
        self.fail_reads = fail_reads
        self.fail_writes = fail_writes

    def get(self, *, ids=None, where=None, include=None):
        if self.fail_reads:
            raise ConnectionError("offline")
        selected = []
        for record_id, row in self.rows.items():
            if ids and record_id not in ids:
                continue
            if where and any(
                row["metadata"].get(key)
                != (condition.get("$eq") if isinstance(condition, dict) else condition)
                for key, condition in where.items()
            ):
                continue
            selected.append((record_id, row))
        return {
            "ids": [item[0] for item in selected],
            "documents": [item[1]["document"] for item in selected],
            "metadatas": [item[1]["metadata"] for item in selected],
        }

    def upsert(self, *, ids, embeddings, documents, metadatas):
        if self.fail_writes:
            raise ConnectionError("offline")
        for record_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.rows[record_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": dict(metadata),
            }

    def update(self, *, ids, metadatas):
        if self.fail_writes:
            raise ConnectionError("offline")
        for record_id, metadata in zip(ids, metadatas, strict=True):
            self.rows[record_id]["metadata"] = dict(metadata)


def _fixture_payload():
    now = datetime.now(UTC)
    evidence = [
        Evidence(
            id="evidence:person",
            provider="demo",
            record_id="P001",
            source_kind="person",
            updated_at=now,
        ),
        Evidence(
            id="evidence:company",
            provider="demo",
            record_id="C001",
            source_kind="company",
            updated_at=now,
        ),
        Evidence(
            id="evidence:edge",
            provider="demo",
            record_id="controls",
            source_kind="relation",
            updated_at=now,
        ),
    ]
    graph = make_graph(
        [
            GraphNode(
                id="person:P001",
                type=NodeType.PERSON,
                label="Elon Musk",
                evidence_ids=["evidence:person"],
            ),
            GraphNode(
                id="company:C001",
                type=NodeType.COMPANY,
                label="Tesla, Inc.",
                evidence_ids=["evidence:company"],
            ),
        ],
        [
            GraphEdge(
                id="relation:controls",
                source="person:P001",
                target="company:C001",
                type=RelationType.CONTROLS,
                label="控制（演示）",
                evidence_ids=["evidence:edge"],
            )
        ],
        "demo-v1",
        evidence,
    )
    signature = QuerySignature(
        intent=Intent.FIND_CONTROLLED_COMPANIES,
        subject_ids=["person:P001"],
        relation_types=[RelationType.CONTROLS],
        requested_relation_types=[RelationType.CONTROLS],
        effective_relation_types=[RelationType.CONTROLS],
        target_types=[NodeType.COMPANY],
        control_policy=ControlQueryPolicy.EXPLICIT_ONLY,
    )
    return signature, graph, evidence


def test_keys_and_embedding_are_deterministic() -> None:
    signature, _, _ = _fixture_payload()
    kwargs = {
        "graph_schema_version": 1,
        "permission_scope": "public-demo",
    }
    # Dataset versions have disjoint canonical namespaces, so a new write cannot
    # overwrite the prior version's otherwise identical signature.
    assert canonical_query_id(signature, data_version="v1", **kwargs) != canonical_query_id(
        signature, data_version="v2", **kwargs
    )
    assert raw_query_hash("  查马斯克控制的公司？ ", "zh-CN", "public-demo") == raw_query_hash(
        "查马斯克控制的公司", "zh-CN", "public-demo"
    )
    assert deterministic_embedding(signature) == deterministic_embedding(signature)
    assert len(deterministic_embedding(signature)) == 64


def test_graph_evidence_catalog_is_complete_deduplicated_and_stable() -> None:
    signature, graph, evidence = _fixture_payload()
    referenced = {
        evidence_id
        for element in [*graph.nodes, *graph.edges]
        for evidence_id in element.evidence_ids
    }
    assert {item.id for item in graph.evidence} == referenced

    duplicate_payload = graph.model_dump()
    newer_retrieval = graph.evidence[0].model_copy(
        update={"retrieved_at": graph.evidence[0].retrieved_at + timedelta(days=1)}
    )
    duplicate_payload["evidence"] = [*graph.evidence, newer_retrieval]
    deduplicated = GraphPayload.model_validate(duplicate_payload)
    assert len(deduplicated.evidence) == len(graph.evidence)
    assert next(
        item for item in deduplicated.evidence if item.id == newer_retrieval.id
    ).retrieved_at == newer_retrieval.retrieved_at

    conflicting_payload = graph.model_dump()
    conflicting_payload["evidence"] = [
        *graph.evidence,
        graph.evidence[0].model_copy(update={"provider": "conflicting-provider"}),
    ]
    with pytest.raises(ValidationError, match="conflicting evidence"):
        GraphPayload.model_validate(conflicting_payload)

    missing_payload = graph.model_dump()
    missing_payload["evidence"] = graph.evidence[1:]
    with pytest.raises(ValidationError, match="missing evidence"):
        GraphPayload.model_validate(missing_payload)

    refreshed = [
        item.model_copy(update={"retrieved_at": item.retrieved_at + timedelta(days=1)})
        for item in evidence
    ]
    refreshed_graph = make_graph(
        graph.nodes, graph.edges, graph.data_version, refreshed
    )
    assert refreshed_graph.graph_id == graph.graph_id

    changed_source = [
        evidence[0].model_copy(update={"provider": "different-provider"}),
        *evidence[1:],
    ]
    changed_graph = make_graph(
        graph.nodes, graph.edges, graph.data_version, changed_source
    )
    assert changed_graph.graph_id != graph.graph_id

    incomplete_graph = graph.model_dump(mode="json")
    incomplete_graph.pop("evidence")
    with pytest.raises(ValidationError):
        CachedPayload.model_validate(
            {
                "answer": "answer",
                "graph": incomplete_graph,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "query_signature": signature.model_dump(mode="json"),
            }
        )


def test_session_merge_preserves_richer_properties_for_stable_ids() -> None:
    _, previous, _ = _fixture_payload()
    company = next(node for node in previous.nodes if node.id == "company:C001")
    richer_company = company.model_copy(
        update={"properties": {"source_id": "C001", "aliases": ["Tesla"], "year": 2003}}
    )
    previous = make_graph(
        [
            richer_company if node.id == richer_company.id else node
            for node in previous.nodes
        ],
        previous.edges,
        previous.data_version,
        previous.evidence,
    )
    partial_company = company.model_copy(update={"properties": {"source_id": "C001"}})
    delta = make_graph(
        [partial_company],
        [],
        previous.data_version,
        [
            item
            for item in previous.evidence
            if item.id in partial_company.evidence_ids
        ],
    )

    merged = merge_graphs(previous, delta)
    merged_company = next(node for node in merged.nodes if node.id == "company:C001")
    assert merged_company.properties == {
        "source_id": "C001",
        "aliases": ["Tesla"],
        "year": 2003,
    }
    assert merged_company.evidence_ids == company.evidence_ids

    conflicting = make_graph(
        [partial_company.model_copy(update={"label": "Different legal identity"})],
        [],
        previous.data_version,
        delta.evidence,
    )
    with pytest.raises(ValueError, match="conflicting node identity"):
        merge_graphs(previous, conflicting)


@pytest.mark.asyncio
async def test_warm_write_raw_hit_and_promotion_without_sliding_expiry() -> None:
    settings = Settings(cache_ttl_hours=1)
    collection = FakeCollection()
    memory = LongTermMemory(settings, data_version="demo-v1", collection=collection)
    signature, graph, evidence = _fixture_payload()

    written = await memory.write(
        raw_query="查马斯克控制的公司",
        locale="zh-CN",
        signature=signature,
        answer="Tesla",
        graph=graph,
        evidence=evidence,
    )
    assert written.success and written.status == CacheStatus.WARM
    expiry = collection.rows[written.record_id]["metadata"]["expires_at_epoch"]

    lookup = await memory.lookup_raw("查马斯克控制的公司", "zh-CN")
    assert lookup.hit and lookup.match_type == "raw_exact"
    promoted = await memory.touch(lookup)
    assert promoted.operation == MemoryOperation.PROMOTE
    metadata = collection.rows[written.record_id]["metadata"]
    assert metadata["status"] == "hot"
    assert metadata["hit_count"] == 1
    assert metadata["expires_at_epoch"] == expiry


@pytest.mark.asyncio
async def test_corrupted_cache_documents_are_marked_stale_and_treated_as_misses() -> None:
    collection = FakeCollection()
    memory = LongTermMemory(Settings(), data_version="demo-v1", collection=collection)
    signature, graph, evidence = _fixture_payload()

    async def fresh_row() -> tuple[str, dict]:
        written = await memory.write(
            raw_query="查马斯克控制的公司",
            locale="zh-CN",
            signature=signature,
            answer="verified answer",
            graph=graph,
            evidence=evidence,
        )
        assert written.success and written.record_id is not None
        return written.record_id, collection.rows[written.record_id]

    record_id, row = await fresh_row()
    document = json.loads(row["document"])
    document["answer"] = "tampered answer"
    row["document"] = json.dumps(document)
    lookup = await memory.lookup_raw("查马斯克控制的公司", "zh-CN")
    assert not lookup.hit and "payload_hash_mismatch" in (lookup.error or "")
    assert collection.rows[record_id]["metadata"]["status"] == "stale"

    record_id, row = await fresh_row()
    document = json.loads(row["document"])
    document["query_signature"]["subject_ids"] = ["person:other"]
    parsed = CachedPayload.model_validate(document)
    row["document"] = json.dumps(document)
    row["metadata"]["payload_hash"] = memory._payload_hash(parsed)
    lookup = await memory.lookup_raw("查马斯克控制的公司", "zh-CN")
    assert not lookup.hit and "canonical_id_mismatch" in (lookup.error or "")
    assert collection.rows[record_id]["metadata"]["status"] == "stale"

    record_id, row = await fresh_row()
    document = json.loads(row["document"])
    document["evidence"][0]["provider"] = "conflicting-provider"
    parsed = CachedPayload.model_validate(document)
    row["document"] = json.dumps(document)
    row["metadata"]["payload_hash"] = memory._payload_hash(parsed)
    lookup = await memory.lookup_raw("查马斯克控制的公司", "zh-CN")
    assert not lookup.hit and "evidence_catalog_mismatch" in (lookup.error or "")
    assert collection.rows[record_id]["metadata"]["status"] == "stale"

    record_id, row = await fresh_row()
    document = json.loads(row["document"])
    document["graph"]["graph_id"] = "graph:tampered"
    parsed = CachedPayload.model_validate(document)
    row["document"] = json.dumps(document)
    row["metadata"]["payload_hash"] = memory._payload_hash(parsed)
    lookup = await memory.lookup_raw("查马斯克控制的公司", "zh-CN")
    assert not lookup.hit and "graph_id_mismatch" in (lookup.error or "")
    assert collection.rows[record_id]["metadata"]["status"] == "stale"


@pytest.mark.asyncio
async def test_conversation_scoped_cache_has_no_globally_reusable_raw_alias() -> None:
    collection = FakeCollection()
    memory = LongTermMemory(Settings(), data_version="demo-v1", collection=collection)
    signature, graph, evidence = _fixture_payload()
    written = await memory.write(
        raw_query="follow-up",
        locale="zh-CN",
        signature=signature,
        answer="answer",
        graph=graph,
        evidence=evidence,
        cache_scope=CacheScope.CONVERSATION,
    )
    assert written.success
    assert "raw_query_hash" not in collection.rows[written.record_id]["metadata"]
    assert not (await memory.lookup_raw("follow-up", "zh-CN")).hit
    assert (await memory.lookup_canonical(signature)).hit


@pytest.mark.asyncio
async def test_ttl_and_version_invalidation_mark_entries_stale() -> None:
    settings = Settings()
    collection = FakeCollection()
    memory = LongTermMemory(settings, data_version="demo-v1", collection=collection)
    signature, graph, evidence = _fixture_payload()
    written = await memory.write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="a",
        graph=graph,
        evidence=evidence,
    )
    collection.rows[written.record_id]["metadata"]["expires_at_epoch"] = 0
    expired = await memory.lookup_canonical(signature)
    assert not expired.hit and expired.error == "cache_expired"
    assert collection.rows[written.record_id]["metadata"]["status"] == "stale"

    # Re-create the row. Canonical lookup uses a disjoint versioned ID, while a
    # legacy raw alias can still locate and retire the old metadata safely.
    await memory.write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="a",
        graph=graph,
        evidence=evidence,
    )
    newer = LongTermMemory(settings, data_version="demo-v2", collection=collection)
    canonical_miss = await newer.lookup_canonical(signature)
    assert not canonical_miss.hit and canonical_miss.error is None
    mismatch = await newer.lookup_raw("q", "zh-CN")
    assert not mismatch.hit and mismatch.error == "data_version_mismatch"

    new_graph = make_graph(graph.nodes, graph.edges, "demo-v2", graph.evidence)
    new_write = await newer.write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="new version",
        graph=new_graph,
        evidence=new_graph.evidence,
    )
    assert new_write.success and new_write.record_id != written.record_id
    recovered_raw = await newer.lookup_raw("q", "zh-CN")
    assert recovered_raw.hit and recovered_raw.record_id == new_write.record_id
    assert recovered_raw.payload and recovered_raw.payload.answer == "new version"

    await memory.write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="a",
        graph=graph,
        evidence=evidence,
    )
    collection.rows[written.record_id]["metadata"]["graph_schema_version"] = "invalid"
    malformed = await memory.lookup_canonical(signature)
    assert not malformed.hit and malformed.error == "graph_schema_version_mismatch"
    assert collection.rows[written.record_id]["metadata"]["status"] == "stale"


@pytest.mark.asyncio
async def test_cache_backend_failures_become_safe_misses() -> None:
    settings = Settings()
    memory = LongTermMemory(
        settings, data_version="demo-v1", collection=FakeCollection(fail_reads=True)
    )
    lookup = await memory.lookup_raw("query", "zh-CN")
    assert not lookup.hit and "offline" in (lookup.error or "")

    failing_write = LongTermMemory(
        settings, data_version="demo-v1", collection=FakeCollection(fail_writes=True)
    )
    signature, graph, evidence = _fixture_payload()
    result = await failing_write.write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="a",
        graph=graph,
        evidence=evidence,
    )
    assert not result.success and result.operation == MemoryOperation.SKIP
    assert result.reason and result.reason.startswith("memory_write_failed")

    invalid_graph = graph.model_copy(update={"graph_id": "graph:invalid"})
    rejected = await LongTermMemory(
        settings, data_version="demo-v1", collection=FakeCollection()
    ).write(
        raw_query="q",
        locale="zh-CN",
        signature=signature,
        answer="a",
        graph=invalid_graph,
        evidence=evidence,
    )
    assert not rejected.success
    assert rejected.reason == "invalid_cache_write:payload_graph_id_mismatch"


def test_write_gate_rejects_missing_evidence_and_accepts_verified_graph() -> None:
    signature, graph, evidence = _fixture_payload()
    base = {
        "cache_hit": False,
        "run_status": "success",
        "research_complete": True,
        "tool_call_count": 2,
        "selected_record_ids": [
            *(node.id for node in graph.nodes),
            *(edge.id for edge in graph.edges),
        ],
        "tool_errors": [],
        "query_result_graph": graph,
        "query_signature": signature,
        "answer": "answer",
        "tool_evidence": evidence,
    }
    assert decide_memory_write(base).operation == MemoryOperation.ADD
    assert decide_memory_write({**base, "tool_evidence": []}).reason == "incomplete_evidence"
    assert decide_memory_write({**base, "tool_errors": [object()]}).reason == "tool_error"
    assert decide_memory_write({**base, "llm_errors": ["failed"]}).reason == "model_error"
    assert (
        decide_memory_write(
            {
                **base,
                "replan_count": 1,
                "replan_reasons": ["A verified alternate route was used."],
            }
        ).operation
        == MemoryOperation.ADD
    )
    assert (
        decide_memory_write({**base, "selected_record_ids": []}).reason
        == "missing_verified_tool_selection"
    )
