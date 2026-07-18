"""Deterministic normalization and identifier helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"[，,。.!！?？;；:：]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", normalized, flags=re.UNICODE)
    return normalized.strip("-") or stable_hash(value)[:12]
