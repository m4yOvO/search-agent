"""Deterministic query keys and embeddings used by the Chroma cache."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from app.ids import canonical_json, normalize_query, stable_hash
from app.schemas import QuerySignature


def raw_query_hash(query: str, locale: str, permission_scope: str) -> str:
    """Build a public-scope exact-query hash.

    Context-dependent results are never written with this alias. That decision is
    made from the Planner's typed cache scope, not from language-specific keywords.
    """

    return stable_hash(
        {
            "query": normalize_query(query),
            "locale": locale.casefold(),
            "permission_scope": permission_scope,
        }
    )


def canonical_query_id(
    signature: QuerySignature,
    *,
    data_version: str,
    graph_schema_version: int,
    permission_scope: str,
    conversation_owner_hash: str | None = None,
) -> str:
    payload = {
        "signature": signature.model_dump(mode="json", exclude_none=True),
        "data_version": data_version,
        "graph_schema_version": graph_schema_version,
        "permission_scope": permission_scope,
    }
    # Context-free IDs intentionally retain their historical shape.  A
    # conversation-scoped result receives an opaque owner binding so an otherwise
    # identical signature cannot overwrite or hydrate another conversation's row.
    if conversation_owner_hash is not None:
        payload["conversation_owner_hash"] = conversation_owner_hash
    return f"canonical:{stable_hash(payload)}"


def deterministic_embedding(value: str | QuerySignature, dimensions: int = 64) -> list[float]:
    """Produce a stable local embedding without downloading a model.

    It is deliberately not used for semantic matching in the MVP.  Storing an
    explicit vector keeps Chroma deterministic and leaves a migration path for
    future calibrated semantic retrieval.
    """

    if dimensions < 8:
        raise ValueError("embedding dimensions must be at least 8")
    text = (
        canonical_json(value.model_dump(mode="json", exclude_none=True))
        if isinstance(value, QuerySignature)
        else normalize_query(value)
    )
    vector: list[float] = []
    counter = 0
    while len(vector) < dimensions:
        block = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        vector.extend((byte / 127.5) - 1.0 for byte in block)
        counter += 1
    vector = vector[:dimensions]
    norm = math.sqrt(sum(component * component for component in vector)) or 1.0
    return [component / norm for component in vector]


def stable_unique(values: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving their first-seen order."""

    return list(dict.fromkeys(values))
