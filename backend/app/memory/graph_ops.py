"""Canonical graph validation and stable session-graph merging."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from app.ids import stable_hash
from app.schemas import Evidence, GraphEdge, GraphNode, GraphPayload


def stable_evidence_rows(evidence: Iterable[Evidence]) -> list[dict[str, Any]]:
    """Return provenance content that defines stable graph identity.

    ``retrieved_at`` describes this particular fetch rather than the underlying
    fact, so it is intentionally excluded.  A source update, provider change, or
    URL change still produces a new graph identity.
    """

    return sorted(
        (
            item.model_dump(mode="json", exclude={"retrieved_at"})
            for item in evidence
        ),
        key=lambda item: item["id"],
    )


def graph_id_for(
    nodes: Iterable[GraphNode],
    edges: Iterable[GraphEdge],
    data_version: str,
    evidence: Iterable[Evidence] = (),
) -> str:
    node_rows = sorted(
        (node.model_dump(mode="json") for node in nodes), key=lambda item: item["id"]
    )
    edge_rows = sorted(
        (edge.model_dump(mode="json") for edge in edges), key=lambda item: item["id"]
    )
    evidence_rows = stable_evidence_rows(evidence)
    return f"graph:{stable_hash({'nodes': node_rows, 'edges': edge_rows, 'evidence': evidence_rows, 'data_version': data_version})}"


def make_graph(
    nodes: Iterable[GraphNode | dict[str, Any]],
    edges: Iterable[GraphEdge | dict[str, Any]],
    data_version: str,
    evidence: Iterable[Evidence | dict[str, Any]] = (),
) -> GraphPayload:
    parsed_nodes = [GraphNode.model_validate(node) for node in nodes]
    parsed_edges = [GraphEdge.model_validate(edge) for edge in edges]
    parsed_evidence = [Evidence.model_validate(item) for item in evidence]
    parsed_nodes = sorted({item.id: item for item in parsed_nodes}.values(), key=lambda item: item.id)
    parsed_edges = sorted({item.id: item for item in parsed_edges}.values(), key=lambda item: item.id)
    referenced_evidence_ids = {
        evidence_id
        for element in [*parsed_nodes, *parsed_edges]
        for evidence_id in element.evidence_ids
    }
    parsed_evidence = [
        item for item in parsed_evidence if item.id in referenced_evidence_ids
    ]
    return GraphPayload(
        graph_id=graph_id_for(
            parsed_nodes, parsed_edges, data_version, parsed_evidence
        ),
        nodes=parsed_nodes,
        edges=parsed_edges,
        evidence=parsed_evidence,
        data_version=data_version,
    )


def empty_graph(data_version: str) -> GraphPayload:
    return make_graph([], [], data_version)


def merge_graphs(previous: GraphPayload | None, delta: GraphPayload) -> GraphPayload:
    if previous is None or previous.data_version != delta.data_version:
        return make_graph(
            delta.nodes, delta.edges, delta.data_version, delta.evidence
        )
    nodes = {node.id: node for node in previous.nodes}
    for incoming in delta.nodes:
        existing = nodes.get(incoming.id)
        if existing is None:
            nodes[incoming.id] = incoming
            continue
        if (existing.type, existing.label) != (incoming.type, incoming.label):
            raise ValueError(f"conflicting node identity for stable ID: {incoming.id}")
        nodes[incoming.id] = existing.model_copy(
            update={
                "properties": {**existing.properties, **incoming.properties},
                "evidence_ids": list(
                    dict.fromkeys([*existing.evidence_ids, *incoming.evidence_ids])
                ),
            }
        )

    edges = {edge.id: edge for edge in previous.edges}
    for incoming in delta.edges:
        existing = edges.get(incoming.id)
        if existing is None:
            edges[incoming.id] = incoming
            continue
        if (
            existing.source,
            existing.target,
            existing.type,
            existing.label,
        ) != (
            incoming.source,
            incoming.target,
            incoming.type,
            incoming.label,
        ):
            raise ValueError(f"conflicting edge identity for stable ID: {incoming.id}")
        edges[incoming.id] = existing.model_copy(
            update={
                "properties": {**existing.properties, **incoming.properties},
                "evidence_ids": list(
                    dict.fromkeys([*existing.evidence_ids, *incoming.evidence_ids])
                ),
            }
        )

    evidence = {item.id: item for item in previous.evidence}
    for incoming in delta.evidence:
        existing = evidence.get(incoming.id)
        if existing is not None:
            existing_stable = existing.model_dump(exclude={"retrieved_at"})
            incoming_stable = incoming.model_dump(exclude={"retrieved_at"})
            if existing_stable != incoming_stable:
                raise ValueError(f"conflicting evidence identity for stable ID: {incoming.id}")
            if existing.retrieved_at >= incoming.retrieved_at:
                continue
        evidence[incoming.id] = incoming
    return make_graph(
        nodes.values(),
        edges.values(),
        delta.data_version,
        evidence.values(),
    )


def evidence_coverage(
    graph: GraphPayload, evidence: Iterable[Evidence] | None = None
) -> float:
    evidence_ids = {
        item.id for item in (graph.evidence if evidence is None else evidence)
    }
    elements = [*graph.nodes, *graph.edges]
    if not elements:
        return 0.0
    supported = sum(
        bool(element.evidence_ids)
        and all(evidence_id in evidence_ids for evidence_id in element.evidence_ids)
        for element in elements
    )
    return supported / len(elements)


def graph_is_valid(graph: GraphPayload | dict[str, Any] | None) -> bool:
    if graph is None:
        return False
    try:
        GraphPayload.model_validate(graph)
    except ValidationError:
        return False
    return True
