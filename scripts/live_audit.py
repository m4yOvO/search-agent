#!/usr/bin/env python3
"""Paid, repeatable Docker/OpenAI audit for the public MVP paths.

The default command recreates the backend with a run-unique Chroma collection,
waits for readiness, and then exercises the public HTTP API.  It deliberately
does not import application code: this is an outside-in audit of the image that
Docker actually serves.

No assertion depends on an exact model-authored answer.  Assertions are made on
status, verified graph records, raw-relation provenance, cache metadata, and the
bounded public execution trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_FILES = ("person 1.json", "company 1.json", "relations 1.json")

RAW_TO_TYPED_RELATION = {
    "CEO_of": "works_at",
    "Chairman_of": "works_at",
    "Chairwoman_of": "works_at",
    "Former_CEO_of": "works_at",
    "Former_Chairman_of": "works_at",
    "Former_President_of": "works_at",
    "Founder_of": "founded",
    "Co-founder_of": "founded",
    "Headquartered_in": "headquartered_in",
    "Owns": "owns",
    "Partner_with": "partner_of",
    "Supplier_to": "supplier_to",
    "Invested_in": "invested_in",
    "Competes_with": "related_to",
    "Uses_AI_from": "related_to",
}
BUSINESS_RELATIONS = frozenset(RAW_TO_TYPED_RELATION) - {"Headquartered_in"}
CONTROL_FALLBACK_RELATIONS = frozenset(
    {
        "Founder_of",
        "Co-founder_of",
        "CEO_of",
        "Chairman_of",
        "Chairwoman_of",
        "Owns",
    }
)
REMOVED_DEMO_DISCLAIMER = "结果来自本地演示数据，不代表实时工商或法律结论。"
ZH_CONTROL_DISCLOSURE = (
    "原始数据没有显式控制记录，以下为创办、现任管理或明确持有关系，"
    "不等同法律控制。"
)
MAX_RESEARCHER_MODEL_ACTIONS = 20
MAX_EXECUTED_TOOL_CALLS = 10
MAX_REPLANS = 2
LIMIT_ERROR_CODES = frozenset({"research_step_limit", "tool_call_limit"})
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTHORIZATION",
)


def _redact_sensitive(
    value: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Remove credential-shaped text before it reaches stderr or a report."""

    redacted = value
    env = os.environ if environment is None else environment
    sensitive_values = sorted(
        {
            str(secret)
            for name, secret in env.items()
            if secret
            and len(str(secret)) >= 4
            and any(marker in name.upper() for marker in SENSITIVE_ENV_MARKERS)
        },
        key=len,
        reverse=True,
    )
    for secret in sensitive_values:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", redacted)
    redacted = re.sub(
        r'(?i)(["\']?(?:[A-Za-z0-9_]*api[_-]?key|[A-Za-z0-9_]*(?:token|secret|password|credential))["\']?\s*[=:]\s*["\']?)[^\s,;"\']+',
        r"\1[REDACTED]",
        redacted,
    )
    return redacted


def _safe_json(value: Any, environment: Mapping[str, str] | None = None) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    return _redact_sensitive(rendered, environment)


class AuditFailure(AssertionError):
    """A stable, user-facing audit assertion failure."""


@dataclass(frozen=True, order=True)
class ExpectedEdge:
    source: str
    target: str
    relation_type: str
    raw_relation: str
    source_row: int


@dataclass(frozen=True)
class AuditQueries:
    multi_entity: str
    locations_followup: str
    multi_entity_paraphrase: str
    ma_yun_founded: str
    ma_yun_founded_paraphrase: str
    ma_yun_control: str
    ma_yun_owns: str
    alibaba_owns: str
    ma_huateng_founded: str

    def items(self) -> list[tuple[str, str]]:
        return [
            ("catalog_aligned_multi_entity", self.multi_entity),
            ("multi_entity_locations", self.locations_followup),
            ("multi_entity_raw_repeat", self.multi_entity),
            ("multi_entity_paraphrase", self.multi_entity_paraphrase),
            ("ma_yun_founded", self.ma_yun_founded),
            ("ma_yun_founded_paraphrase", self.ma_yun_founded_paraphrase),
            ("ma_yun_control", self.ma_yun_control),
            ("ma_yun_owns", self.ma_yun_owns),
            ("alibaba_owns", self.alibaba_owns),
            ("ma_huateng_founded", self.ma_huateng_founded),
        ]


@dataclass(frozen=True)
class RawDataset:
    persons: tuple[dict[str, Any], ...]
    companies: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    token_to_id: Mapping[str, str]
    label_by_id: Mapping[str, str]

    @classmethod
    def load(cls, directory: Path) -> "RawDataset":
        actual_json = {path.name for path in directory.glob("*.json")}
        if actual_json != set(RAW_FILES):
            raise AuditFailure(
                f"data/ must contain exactly {sorted(RAW_FILES)}; got {sorted(actual_json)}"
            )

        persons = tuple(_load_json_array(directory / RAW_FILES[0]))
        companies = tuple(_load_json_array(directory / RAW_FILES[1]))
        relations = tuple(_load_json_array(directory / RAW_FILES[2]))
        if (len(persons), len(companies), len(relations)) != (20, 30, 109):
            raise AuditFailure(
                "raw fixture row counts must remain 20 persons, 30 companies, "
                f"and 109 relations; got {(len(persons), len(companies), len(relations))}"
            )

        _validate_rows(persons, {"id", "name", "nationality", "summary"}, "person")
        _validate_rows(
            companies,
            {"id", "name", "legal_rep_id", "city", "founded_year"},
            "company",
        )
        _validate_rows(relations, {"head", "relation", "tail"}, "relation")

        token_to_id: dict[str, str] = {}
        label_by_id: dict[str, str] = {}
        for namespace, rows in (("person", persons), ("company", companies)):
            for row in rows:
                raw_id = str(row["id"])
                stable_id = f"{namespace}:{raw_id}"
                raw_name = str(row["name"])
                for token in (raw_id, raw_name):
                    previous = token_to_id.setdefault(token, stable_id)
                    if previous != stable_id:
                        raise AuditFailure(f"raw endpoint token is ambiguous: {token!r}")
                label_by_id[stable_id] = raw_name

        unknown_relations = {
            str(row["relation"]) for row in relations
        } - RAW_TO_TYPED_RELATION.keys()
        if unknown_relations:
            raise AuditFailure(
                f"audit mapping is missing raw relation types: {sorted(unknown_relations)}"
            )

        return cls(
            persons=persons,
            companies=companies,
            relations=relations,
            token_to_id=token_to_id,
            label_by_id=label_by_id,
        )

    def require_label(self, stable_id: str, expected: str) -> None:
        actual = self.label_by_id.get(stable_id)
        if actual != expected:
            raise AuditFailure(
                f"{stable_id} must retain raw label {expected!r}; got {actual!r}"
            )

    def resolve_endpoint(self, raw_value: str) -> str:
        entity_id = self.token_to_id.get(raw_value)
        if entity_id is None:
            raise AuditFailure(
                f"audit expected a person/company endpoint but raw value is unresolved: {raw_value!r}"
            )
        return entity_id

    def relation_edges(
        self,
        *,
        subject_ids: Iterable[str],
        raw_relation_types: Iterable[str],
        direction: str = "any",
    ) -> set[ExpectedEdge]:
        subjects = set(subject_ids)
        raw_types = set(raw_relation_types)
        result: set[ExpectedEdge] = set()
        for row_number, row in enumerate(self.relations, start=1):
            raw_relation = str(row["relation"])
            if raw_relation not in raw_types:
                continue
            # This audit only derives expectations for entity-to-entity relations.
            source = self.token_to_id.get(str(row["head"]))
            target = self.token_to_id.get(str(row["tail"]))
            if source is None or target is None:
                continue
            matches = {
                "outgoing": source in subjects,
                "incoming": target in subjects,
                "any": source in subjects or target in subjects,
            }
            if direction not in matches:
                raise AuditFailure(f"unsupported audit direction: {direction}")
            if matches[direction]:
                result.add(
                    ExpectedEdge(
                        source=source,
                        target=target,
                        relation_type=RAW_TO_TYPED_RELATION[raw_relation],
                        raw_relation=raw_relation,
                        source_row=row_number,
                    )
                )
        return result

    def headquarters_edges(self, company_ids: Iterable[str]) -> set[ExpectedEdge]:
        wanted = set(company_ids)
        result: set[ExpectedEdge] = set()
        for row_number, row in enumerate(self.relations, start=1):
            if row["relation"] != "Headquartered_in":
                continue
            source = self.resolve_endpoint(str(row["head"]))
            if source not in wanted:
                continue
            city = str(row["tail"])
            result.add(
                ExpectedEdge(
                    source=source,
                    target=f"location:{_slugify(city)}",
                    relation_type="headquartered_in",
                    raw_relation="Headquartered_in",
                    source_row=row_number,
                )
            )
        return result

    def raw_relation(self, source_row: int) -> Mapping[str, Any]:
        if not 1 <= source_row <= len(self.relations):
            raise AuditFailure(f"raw relation source_row is out of range: {source_row}")
        return self.relations[source_row - 1]

    def location_nodes_for_edges(
        self, edges: Iterable[ExpectedEdge]
    ) -> dict[str, tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        for edge in edges:
            if edge.raw_relation != "Headquartered_in":
                continue
            city = str(self.raw_relation(edge.source_row)["tail"])
            result[edge.target] = ("location", city)
        return result


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read raw fixture {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AuditFailure(f"raw fixture must be an array of objects: {path}")
    return value


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], required: set[str], kind: str
) -> None:
    for row_number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise AuditFailure(
                f"{kind} row {row_number} must preserve fields {sorted(required)}; "
                f"got {sorted(row)}"
            )


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unknown"


def build_queries(dataset: RawDataset) -> AuditQueries:
    """Generate the audit questions only after checking their raw-data anchors."""

    dataset.require_label("person:P001", "Elon Musk")
    dataset.require_label("company:C001", "Tesla, Inc.")
    dataset.require_label("person:P004", "马云")
    dataset.require_label("person:P005", "马化腾")
    dataset.require_label("company:C005", "阿里巴巴集团")
    dataset.require_label("company:C023", "阿里云")
    return AuditQueries(
        multi_entity="特斯拉和马斯克有哪些关联公司？",
        locations_followup="这些公司在哪？",
        multi_entity_paraphrase="请列出与马斯克或特斯拉存在一跳业务关联的公司。",
        ma_yun_founded="马云创办了哪些公司？",
        ma_yun_founded_paraphrase="哪些企业是马云成立的？",
        ma_yun_control="马云控制了哪些公司？",
        ma_yun_owns="马云拥有哪些公司？",
        alibaba_owns="阿里巴巴集团拥有哪些公司？",
        ma_huateng_founded="马化腾创办了哪些公司？",
    )


class ApiClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def chat(self, *, message: str, conversation_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/chat",
            {
                "conversation_id": conversation_id,
                "message": message,
                "locale": "zh-CN",
            },
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _redact_sensitive(
                exc.read().decode("utf-8", errors="replace")[:2_000]
            )
            raise AuditFailure(
                _redact_sensitive(
                    f"{method} {path} returned HTTP {exc.code}: {detail}"
                )
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AuditFailure(
                _redact_sensitive(f"{method} {path} failed: {exc}")
            ) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditFailure(f"{method} {path} returned non-JSON data") from exc
        if not isinstance(parsed, dict):
            raise AuditFailure(f"{method} {path} did not return a JSON object")
        return parsed


class DockerStack:
    def __init__(self, *, namespace: str, build: bool) -> None:
        self.namespace = namespace
        self.build = build
        self.environment = {
            **os.environ,
            "CHROMA_COLLECTION_PREFIX": namespace,
        }

    def start(self) -> None:
        self._run(["docker", "compose", "config", "--quiet"])
        command = ["docker", "compose", "up"]
        if self.build:
            command.append("--build")
        command.extend(["-d", "--force-recreate", "backend", "frontend"])
        self._run(command)

    def logs(self, tail: int = 2_000) -> str:
        completed = self._run(
            [
                "docker",
                "compose",
                "logs",
                "--no-color",
                f"--tail={tail}",
                "backend",
            ]
        )
        return completed.stdout

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self.environment,
                text=True,
                # Compose output can echo expanded environment values on error.
                # Always capture it and expose only a redacted tail below.
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise AuditFailure(f"required executable is unavailable: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = _redact_sensitive(
                (exc.stderr or exc.stdout or "").strip()[-4_000:],
                self.environment,
            )
            raise AuditFailure(
                _redact_sensitive(
                    f"command failed ({' '.join(command)}): {detail or exc.returncode}",
                    self.environment,
                )
            ) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _graph(body: Mapping[str, Any]) -> Mapping[str, Any]:
    graph = body.get("graph")
    _require(isinstance(graph, dict), "response graph must be an object")
    return graph


def _nodes(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nodes = _graph(body).get("nodes")
    _require(isinstance(nodes, list), "response graph nodes must be an array")
    _require(all(isinstance(node, dict) for node in nodes), "graph node must be an object")
    return nodes


def _edges(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    edges = _graph(body).get("edges")
    _require(isinstance(edges, list), "response graph edges must be an array")
    _require(all(isinstance(edge, dict) for edge in edges), "graph edge must be an object")
    return edges


def edge_signatures(body: Mapping[str, Any]) -> set[ExpectedEdge]:
    signatures: list[ExpectedEdge] = []
    for edge in _edges(body):
        properties = edge.get("properties")
        _require(isinstance(properties, dict), f"edge {edge.get('id')} lacks properties")
        source_row = properties.get("source_row")
        _require(
            isinstance(source_row, int) and source_row > 0,
            f"edge {edge.get('id')} lacks one-based raw source_row",
        )
        raw_relation = properties.get("raw_relation")
        _require(
            isinstance(raw_relation, str) and raw_relation,
            f"edge {edge.get('id')} lacks raw_relation",
        )
        signatures.append(
            ExpectedEdge(
                source=str(edge.get("source")),
                target=str(edge.get("target")),
                relation_type=str(edge.get("type")),
                raw_relation=raw_relation,
                source_row=source_row,
            )
        )
    _require(
        len(signatures) == len(set(signatures)),
        "graph contains duplicate edge provenance signatures",
    )
    return set(signatures)


def assert_success(body: Mapping[str, Any], case: str) -> None:
    _require(body.get("status") == "success", f"{case}: status is not success")
    _require(body.get("error_code") is None, f"{case}: unexpected error_code")
    _require(isinstance(body.get("request_id"), str), f"{case}: missing request_id")
    _require(isinstance(body.get("conversation_id"), str), f"{case}: missing conversation_id")
    _require(body.get("disclaimer") == "", f"{case}: disclaimer must be empty")
    _require(
        REMOVED_DEMO_DISCLAIMER not in str(body.get("answer", "")),
        f"{case}: removed demo disclaimer is still present in the answer",
    )
    assert_graph_integrity(body, case)


def assert_graph_integrity(body: Mapping[str, Any], case: str) -> None:
    graph = _graph(body)
    evidence = graph.get("evidence")
    _require(isinstance(evidence, list), f"{case}: graph evidence must be an array")
    nodes = _nodes(body)
    edges = _edges(body)
    node_ids = [str(node.get("id")) for node in nodes]
    edge_ids = [str(edge.get("id")) for edge in edges]
    raw_evidence_ids = [
        str(item.get("id"))
        for item in evidence
        if isinstance(item, dict) and item.get("id")
    ]
    _require(len(node_ids) == len(set(node_ids)), f"{case}: duplicate node IDs")
    _require(len(edge_ids) == len(set(edge_ids)), f"{case}: duplicate edge IDs")
    _require(
        len(raw_evidence_ids) == len(evidence),
        f"{case}: malformed evidence catalog item",
    )
    _require(
        len(raw_evidence_ids) == len(set(raw_evidence_ids)),
        f"{case}: duplicate evidence IDs",
    )
    evidence_ids = set(raw_evidence_ids)
    elements = [*nodes, *edges]
    referenced_evidence_ids: set[str] = set()
    for element in elements:
        referenced = element.get("evidence_ids")
        _require(
            isinstance(referenced, list) and bool(referenced),
            f"{case}: graph element {element.get('id')} lacks evidence IDs",
        )
        _require(
            set(referenced) <= evidence_ids,
            f"{case}: graph element {element.get('id')} references missing evidence",
        )
        _require(
            len(referenced) == len(set(referenced)),
            f"{case}: graph element {element.get('id')} repeats an evidence ID",
        )
        referenced_evidence_ids.update(str(item) for item in referenced)
    if elements:
        _require(evidence_ids, f"{case}: non-empty graph lacks evidence catalog")
    _require(
        referenced_evidence_ids == evidence_ids,
        f"{case}: graph evidence catalog is not exact; "
        f"unreferenced={sorted(evidence_ids - referenced_evidence_ids)}, "
        f"missing={sorted(referenced_evidence_ids - evidence_ids)}",
    )
    # Validate provenance before converting signatures to a set, so duplicate
    # source rows cannot be hidden by set semantics.
    edge_signatures(body)


def assert_exact_nodes(
    body: Mapping[str, Any],
    expected: Mapping[str, tuple[str, str]],
    case: str,
) -> None:
    actual = {
        str(node.get("id")): (str(node.get("type")), str(node.get("label")))
        for node in _nodes(body)
    }
    _require(
        actual == dict(expected),
        f"{case}: node mismatch; missing={sorted(set(expected) - set(actual))}, "
        f"unexpected={sorted(set(actual) - set(expected))}",
    )


def assert_raw_provenance(
    body: Mapping[str, Any], dataset: RawDataset, case: str
) -> None:
    evidence_by_id = {
        str(item["id"]): item
        for item in _graph(body).get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    _require(
        all(
            item.get("provider") == "local-raw-json-mock"
            and item.get("is_demo") is True
            for item in evidence_by_id.values()
        ),
        f"{case}: Evidence provider/demo boundary drift",
    )
    for edge in _edges(body):
        properties = edge.get("properties")
        _require(isinstance(properties, dict), f"{case}: edge properties missing")
        row_number = properties.get("source_row")
        _require(isinstance(row_number, int), f"{case}: edge source_row missing")
        raw = dataset.raw_relation(row_number)
        expected_source = (
            dataset.token_to_id.get(str(raw["head"]))
            or f"location:{_slugify(str(raw['head']))}"
        )
        expected_target = (
            dataset.token_to_id.get(str(raw["tail"]))
            or f"location:{_slugify(str(raw['tail']))}"
        )
        _require(edge.get("source") == expected_source, f"{case}: raw head projection drift")
        _require(edge.get("target") == expected_target, f"{case}: raw tail projection drift")
        _require(edge.get("label") == raw["relation"], f"{case}: edge label changed raw relation")
        _require(
            properties.get("raw_head") == raw["head"]
            and properties.get("raw_relation") == raw["relation"]
            and properties.get("raw_tail") == raw["tail"],
            f"{case}: edge raw head/relation/tail provenance drift at row {row_number}",
        )
        _require(
            properties.get("source_file") == "relations 1.json",
            f"{case}: edge source_file is not relations 1.json",
        )
        relation_evidence = [
            evidence_by_id.get(str(evidence_id))
            for evidence_id in edge.get("evidence_ids", [])
        ]
        _require(
            any(
                item
                and item.get("source_kind") == "raw_relation"
                and item.get("record_id") == f"relations 1.json#{row_number}"
                for item in relation_evidence
            ),
            f"{case}: edge row {row_number} lacks exact raw-relation Evidence",
        )

    for node in _nodes(body):
        entity_id = str(node.get("id"))
        if entity_id in dataset.label_by_id:
            _require(
                node.get("label") == dataset.label_by_id[entity_id],
                f"{case}: raw entity label changed for {entity_id}",
            )
            properties = node.get("properties")
            _require(isinstance(properties, dict), f"{case}: node properties missing")
            namespace = entity_id.partition(":")[0]
            expected_file = "person 1.json" if namespace == "person" else "company 1.json"
            _require(
                properties.get("source_file") == expected_file,
                f"{case}: entity {entity_id} source_file drift",
            )
            raw_id = entity_id.partition(":")[2]
            expected_kind = "raw_person" if namespace == "person" else "raw_company"
            node_evidence = [
                evidence_by_id.get(str(evidence_id))
                for evidence_id in node.get("evidence_ids", [])
            ]
            _require(
                any(
                    item
                    and item.get("source_kind") == expected_kind
                    and item.get("record_id") == raw_id
                    for item in node_evidence
                ),
                f"{case}: raw entity {entity_id} lacks exact source Evidence",
            )
        elif entity_id.startswith("location:"):
            city = str(node.get("label"))
            _require(
                entity_id == f"location:{_slugify(city)}",
                f"{case}: location ID/label projection drift",
            )
            properties = node.get("properties")
            _require(
                isinstance(properties, dict)
                and properties.get("source_file") == "company 1.json"
                and properties.get("source_id") == city,
                f"{case}: location provenance properties drift",
            )
            company_ids = [
                str(company["id"])
                for company in dataset.companies
                if str(company["city"]) == city
            ]
            location_evidence = [
                evidence_by_id.get(str(evidence_id))
                for evidence_id in node.get("evidence_ids", [])
            ]
            _require(
                any(
                    item
                    and item.get("source_kind") == "raw_company_city"
                    and item.get("record_id") == ",".join(company_ids)
                    for item in location_evidence
                ),
                f"{case}: location {entity_id} lacks exact raw-company-city Evidence",
            )


def assert_exact_edges(
    body: Mapping[str, Any], expected: set[ExpectedEdge], case: str
) -> None:
    actual = edge_signatures(body)
    _require(
        actual == expected,
        f"{case}: edge mismatch; missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def _trace(body: Mapping[str, Any], case: str) -> Mapping[str, Any]:
    trace = body.get("trace")
    _require(isinstance(trace, dict), f"{case}: trace must be an object")
    return trace


def trace_tools(body: Mapping[str, Any], case: str) -> set[str]:
    trace = _trace(body, case)
    steps = trace.get("agent_steps")
    _require(isinstance(steps, list), f"{case}: agent_steps must be an array")
    return {
        str(step["tool"])
        for step in steps
        if isinstance(step, dict) and step.get("tool")
    }


def researcher_relation_scopes(body: Mapping[str, Any], case: str) -> list[set[str]]:
    trace = _trace(body, case)
    steps = trace.get("agent_steps")
    _require(isinstance(steps, list), f"{case}: agent_steps must be an array")
    return [
        {str(value) for value in step.get("relation_types", [])}
        for step in steps
        if isinstance(step, dict)
        and step.get("role") == "researcher"
        and step.get("tool") == "relations"
    ]


def researcher_relation_events(
    body: Mapping[str, Any], case: str
) -> list[Mapping[str, Any]]:
    steps = _trace(body, case).get("agent_steps")
    _require(isinstance(steps, list), f"{case}: agent_steps must be an array")
    return [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("role") == "researcher"
        and step.get("tool") == "relations"
    ]


def assert_catalog_alignment_trace(body: Mapping[str, Any], case: str) -> None:
    steps = _trace(body, case).get("agent_steps")
    _require(isinstance(steps, list), f"{case}: agent_steps must be an array")
    entity_results = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("role") == "researcher"
        and step.get("action") == "tool_result"
        and step.get("tool") in {"persons", "companies"}
    ]
    def matched_entity_ids(
        step: Mapping[str, Any], tool: str
    ) -> tuple[str, ...]:
        record_ids = step.get("record_ids", [])
        _require(
            isinstance(record_ids, list),
            f"{case}: entity tool record_ids must be an array",
        )
        entity_prefix = "company:" if tool == "companies" else "person:"
        allowed_prefixes = (
            ("company:", "location:", "relation:")
            if tool == "companies"
            else ("person:",)
        )
        _require(
            all(
                isinstance(record_id, str)
                and record_id.startswith(allowed_prefixes)
                for record_id in record_ids
            ),
            f"{case}: entity tool trace contains an invalid record kind",
        )
        # Entity lookup receipts may include evidence-closure records (for
        # example a company's raw headquarters edge and location node).  Match
        # proof is about the entity record itself, so compare only that type
        # while retaining the prefix guard above for every attached record.
        return tuple(
            record_id
            for record_id in record_ids
            if record_id.startswith(entity_prefix)
        )

    actual = {
        tool: [
            (
                str(step.get("resolution_strategy")),
                str(step.get("resolution_version")),
                matched_entity_ids(step, tool),
            )
            for step in entity_results
            if step.get("tool") == tool
        ]
        for tool in ("companies", "persons")
    }
    expected = {
        "companies": [
            ("exact", "entity-match-v2", ()),
            ("fuzzy", "entity-match-v2", ()),
            ("cross_language_exact", "entity-match-v2", ("company:C001",)),
        ],
        "persons": [
            ("exact", "entity-match-v2", ()),
            ("fuzzy", "entity-match-v2", ()),
            ("cross_language_exact", "entity-match-v2", ("person:P001",)),
        ],
    }
    _require(
        actual == expected,
        f"{case}: catalog-alignment trace mismatch; expected={expected}, actual={actual}",
    )
    _require(
        all(
            step.get("resolution_strategy") is None
            and step.get("resolution_version") is None
            for step in steps
            if step not in entity_results
        ),
        f"{case}: resolution metadata appeared outside accepted entity tool_result steps",
    )


def assert_planner_result_merge(
    body: Mapping[str, Any], expected: str, case: str
) -> None:
    steps = _trace(body, case).get("agent_steps")
    planner_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("role") == "planner"
        and step.get("action") == "plan"
    ]
    _require(len(planner_steps) == 1, f"{case}: expected exactly one accepted Planner plan")
    _require(
        planner_steps[0].get("result_merge") == expected,
        f"{case}: Planner result_merge is not {expected}",
    )


def _assert_route_shape(
    body: Mapping[str, Any], case: str, *, path: str
) -> None:
    trace = _trace(body, case)
    route = trace.get("route_history")
    _require(isinstance(route, list) and route, f"{case}: route_history missing")
    if path == "raw_hit":
        expected = [
            "begin_turn",
            "raw_cache_probe",
            "cache_hydrate",
            "cache_touch",
            "merge_session_graph",
            "compact_session",
        ]
        _require(route == expected, f"{case}: raw-hit route drift: {route}")
        return

    fresh_prefix = [
        "begin_turn",
        "raw_cache_probe",
        "planner_analyze",
        "planner_tasks",
    ]
    _require(
        route[:4] == fresh_prefix,
        f"{case}: fresh route has an invalid prefix: {route[:4]}",
    )
    researcher_calls = int(trace.get("researcher_model_calls", 0))
    research_segment = route[4 : 4 + researcher_calls * 2]
    _require(
        research_segment == ["researcher", "result_gate"] * researcher_calls,
        f"{case}: Researcher/ResultGate route is not strictly paired",
    )
    tail = route[4 + researcher_calls * 2 :]
    expected_tails = {
        "fresh": [
            "canonical_cache_probe",
            "visualizer",
            "memory_write",
            "merge_session_graph",
            "compact_session",
        ],
        "no_results": [
            "visualizer",
            "memory_write",
            "merge_session_graph",
            "compact_session",
        ],
        "canonical_hit": [
            "canonical_cache_probe",
            "cache_hydrate",
            "cache_touch",
            "merge_session_graph",
            "compact_session",
        ],
    }
    _require(path in expected_tails, f"{case}: unknown route path {path}")
    _require(tail == expected_tails[path], f"{case}: {path} route tail drift: {tail}")


def assert_fresh_trace(
    body: Mapping[str, Any],
    case: str,
    *,
    required_tools: set[str],
    require_visualizer: bool = True,
    route_path: str = "fresh",
) -> None:
    trace = _trace(body, case)
    for field in (
        "model_calls",
        "planner_model_calls",
        "researcher_model_calls",
        "tool_calls",
        "research_steps",
    ):
        _require(
            isinstance(trace.get(field), int) and trace[field] > 0,
            f"{case}: fresh trace requires positive {field}",
        )
    _require(trace.get("researcher_invoked") is True, f"{case}: Researcher not invoked")
    _require(trace.get("model_provider") == "openai", f"{case}: provider is not OpenAI")
    _require(isinstance(trace.get("model_name"), str), f"{case}: model name missing")
    _require(
        trace.get("planner_model_calls") == 2,
        f"{case}: fresh research path must call both Planner stages exactly once",
    )
    _require(trace.get("replans") == 0, f"{case}: unexpected replan")
    _require(
        trace.get("research_steps") == trace.get("researcher_model_calls"),
        f"{case}: research_steps and Researcher model calls diverged",
    )
    _require(
        trace.get("model_calls")
        == trace.get("planner_model_calls")
        + trace.get("researcher_model_calls")
        + trace.get("visualizer_model_calls"),
        f"{case}: aggregate model count does not equal role counts",
    )
    _require(
        trace.get("researcher_model_calls") <= MAX_RESEARCHER_MODEL_ACTIONS,
        f"{case}: Researcher exceeded the absolute action budget",
    )
    _require(
        trace.get("tool_calls") <= MAX_EXECUTED_TOOL_CALLS,
        f"{case}: tool calls exceeded the contract budget",
    )
    roles = {
        str(step.get("role"))
        for step in trace.get("agent_steps", [])
        if isinstance(step, dict)
    }
    required_roles = {"planner", "researcher"}
    if require_visualizer:
        required_roles.add("visualizer")
        _require(
            isinstance(trace.get("visualizer_model_calls"), int)
            and trace["visualizer_model_calls"] > 0,
            f"{case}: Visualizer was not called",
        )
    else:
        _require(
            trace.get("visualizer_model_calls") == 0,
            f"{case}: cache hydration unexpectedly called Visualizer",
        )
    _require(required_roles <= roles, f"{case}: missing Agent trace roles {required_roles - roles}")
    actual_tools = trace_tools(body, case)
    _require(
        required_tools <= actual_tools,
        f"{case}: missing tool traces {required_tools - actual_tools}",
    )
    steps = trace.get("agent_steps", [])
    for index, step in enumerate(steps):
        _require(isinstance(step, dict), f"{case}: malformed Agent step {index}")
        record_ids = step.get("record_ids", [])
        _require(
            isinstance(record_ids, list) and len(record_ids) == len(set(record_ids)),
            f"{case}: Agent step {index} repeats a record ID",
        )
        _require(
            step.get("error_code") is None,
            f"{case}: successful flow contains Agent error {step.get('error_code')}",
        )
    tool_result_count = sum(
        1
        for step in steps
        if step.get("role") == "researcher" and step.get("action") == "tool_result"
    )
    _require(
        tool_result_count == trace.get("tool_calls"),
        f"{case}: executed tool count and tool_result trace count diverged",
    )
    _require(
        not {
            step.get("error_code")
            for step in steps
            if step.get("error_code") in LIMIT_ERROR_CODES
        },
        f"{case}: successful flow hit a budget limit",
    )
    _assert_route_shape(body, case, path=route_path)


def assert_raw_cache_hit(body: Mapping[str, Any], case: str) -> None:
    memory = body.get("memory")
    _require(isinstance(memory, dict), f"{case}: memory must be an object")
    _require(memory.get("cache_hit") is True, f"{case}: raw cache did not hit")
    _require(memory.get("match_type") == "raw_exact", f"{case}: not a raw_exact hit")
    _require(memory.get("status") == "hot", f"{case}: cache was not promoted to HOT")
    _require(
        memory.get("write_operation") == "promote",
        f"{case}: first exact reuse did not report PROMOTE",
    )
    trace = _trace(body, case)
    for field in (
        "model_calls",
        "planner_model_calls",
        "researcher_model_calls",
        "visualizer_model_calls",
        "tool_calls",
    ):
        _require(trace.get(field) == 0, f"{case}: cache hit has non-zero {field}")
    _require(trace.get("researcher_invoked") is False, f"{case}: Researcher ran on cache hit")
    _require(trace.get("agent_steps") == [], f"{case}: cache hit contains Agent steps")
    _require(trace.get("research_steps") == 0, f"{case}: cache hit has research steps")
    _require(trace.get("replans") == 0, f"{case}: cache hit has replans")
    _assert_route_shape(body, case, path="raw_hit")


def assert_warm_add(body: Mapping[str, Any], case: str) -> None:
    memory = body.get("memory")
    _require(isinstance(memory, dict), f"{case}: memory must be an object")
    _require(memory.get("cache_hit") is False, f"{case}: expected a fresh cache miss")
    _require(memory.get("status") == "warm", f"{case}: cache status is not WARM")
    _require(memory.get("write_operation") == "add", f"{case}: cache operation is not ADD")
    _require(isinstance(memory.get("result_id"), str), f"{case}: cache result_id missing")


def assert_no_results_cache_skip(body: Mapping[str, Any], case: str) -> None:
    memory = body.get("memory")
    _require(isinstance(memory, dict), f"{case}: memory must be an object")
    _require(memory.get("cache_hit") is False, f"{case}: no-results unexpectedly hit cache")
    _require(memory.get("status") is None, f"{case}: no-results acquired cache status")
    _require(memory.get("write_operation") == "skip", f"{case}: no-results was not SKIP")
    _require(memory.get("result_id") is None, f"{case}: no-results acquired cache result_id")


def assert_control_public_contract(body: Mapping[str, Any], case: str) -> None:
    """Verify every control invariant observable through the public response.

    Direction and raw-relation qualifiers are intentionally absent from the public
    trace. Their output effect is checked by exact graph provenance; the report
    separately records those two call-time values as not publicly observable.
    """

    answer = str(body.get("answer", ""))
    _require(
        answer.count(ZH_CONTROL_DISCLOSURE) == 1,
        f"{case}: fixed broad-control disclosure is missing or duplicated",
    )
    _require(
        all(edge.relation_type != "controls" for edge in edge_signatures(body)),
        f"{case}: runtime invented a controls edge",
    )
    control_events = researcher_relation_events(body, case)
    control_calls = [event for event in control_events if event.get("action") == "call_tool"]
    control_results = [event for event in control_events if event.get("action") == "tool_result"]
    expected_scopes = [{"controls"}, {"founded", "works_at", "owns"}]
    # Current safe public traces retain executed tool results and fingerprints;
    # some compatible trace versions also retain the preceding call proposal.
    # Validate proposals when present, but do not require a field the public API
    # intentionally omits. The two executed result scopes below are mandatory.
    if control_calls:
        _require(
            [set(event.get("relation_types", [])) for event in control_calls]
            == expected_scopes,
            f"{case}: control phases are missing, out of order, or have the wrong typed scope",
        )
    _require(
        [set(event.get("relation_types", [])) for event in control_results]
        == expected_scopes,
        f"{case}: control tool-result phases are out of order",
    )
    _require(
        control_results[0].get("count") == 0
        and control_results[0].get("record_ids") == [],
        f"{case}: explicit controls phase was not an exhaustive visible zero result",
    )
    _require(
        {"relation:raw:0006", "relation:raw:0106"}
        <= set(control_results[1].get("record_ids", [])),
        f"{case}: fallback result lacks both raw Founder_of records",
    )


def assert_canonical_or_fresh_equivalent_trace(
    body: Mapping[str, Any],
    case: str,
    *,
    required_tools: set[str],
    expected_touch_operation: str,
) -> None:
    """Accept a canonical hit or a separately verified, signature-distinct result.

    Two paraphrases can select the same raw rows while retaining different
    requested raw qualifiers in their canonical signatures.  Those qualifiers
    are deliberately cache-significant, so graph equality alone must not force
    cache reuse.  Either path must still complete full Agent/tool verification;
    only an exact canonical signature may bypass Visualizer.
    """

    memory = body.get("memory")
    _require(isinstance(memory, dict), f"{case}: memory must be an object")
    if memory.get("cache_hit") is False:
        assert_fresh_trace(
            body,
            case,
            required_tools=required_tools,
            require_visualizer=True,
            route_path="fresh",
        )
        assert_warm_add(body, case)
        return

    assert_fresh_trace(
        body,
        case,
        required_tools=required_tools,
        require_visualizer=False,
        route_path="canonical_hit",
    )
    _require(memory.get("cache_hit") is True, f"{case}: invalid cache_hit value")
    _require(
        memory.get("match_type") == "canonical_exact",
        f"{case}: paraphrase did not use canonical_exact",
    )
    _require(memory.get("status") == "hot", f"{case}: canonical hit is not HOT")
    _require(
        memory.get("write_operation") == expected_touch_operation,
        f"{case}: canonical cache operation is not {expected_touch_operation}",
    )


def _new_conversation() -> str:
    return str(uuid4())


def _case_report(
    name: str,
    body: Mapping[str, Any],
    *,
    unobservable_constraints: Sequence[str] = (),
) -> dict[str, Any]:
    memory = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    trace = body.get("trace") if isinstance(body.get("trace"), dict) else {}
    return {
        "case": name,
        "request_id": body.get("request_id"),
        "conversation_id": body.get("conversation_id"),
        "status": body.get("status"),
        "nodes": len(_nodes(body)),
        "edges": len(_edges(body)),
        "cache_hit": memory.get("cache_hit"),
        "cache_match_type": memory.get("match_type"),
        "cache_status": memory.get("status"),
        "model_calls": trace.get("model_calls"),
        "tool_calls": trace.get("tool_calls"),
        "planner_calls": trace.get("planner_model_calls"),
        "researcher_calls": trace.get("researcher_model_calls"),
        "visualizer_calls": trace.get("visualizer_model_calls"),
        "research_steps": trace.get("research_steps"),
        "replans": trace.get("replans"),
        "route_history": list(trace.get("route_history", [])),
        "unobservable_constraints": list(unobservable_constraints),
    }


def run_live_audit(client: ApiClient, dataset: RawDataset) -> list[dict[str, Any]]:
    queries = build_queries(dataset)
    reports: list[dict[str, Any]] = []

    association_edges = dataset.relation_edges(
        subject_ids={"person:P001", "company:C001"},
        raw_relation_types=BUSINESS_RELATIONS,
        direction="any",
    )
    association_company_ids = {
        endpoint
        for edge in association_edges
        for endpoint in (edge.source, edge.target)
        if endpoint.startswith("company:")
    }
    # The union is computed per subject before merging: C001 is P001's
    # opposite company endpoint even though C001 is also the other explicit seed.
    # It therefore remains part of the complete company focus for “这些公司”.
    followup_company_ids = association_company_ids
    headquarters_edges = dataset.headquarters_edges(followup_company_ids)
    association_nodes = {
        entity_id: ("company", dataset.label_by_id[entity_id])
        for entity_id in association_company_ids
    } | {"person:P001": ("person", "Elon Musk")}
    location_session_nodes = association_nodes | dataset.location_nodes_for_edges(
        headquarters_edges
    )

    primary_conversation = _new_conversation()
    primary = client.chat(
        message=queries.multi_entity,
        conversation_id=primary_conversation,
    )
    case = "catalog_aligned_multi_entity"
    assert_success(primary, case)
    assert_exact_edges(primary, association_edges, case)
    assert_exact_nodes(primary, association_nodes, case)
    assert_raw_provenance(primary, dataset, case)
    assert_fresh_trace(
        primary,
        case,
        required_tools={"companies", "persons", "relations"},
    )
    assert_planner_result_merge(primary, "union", case)
    assert_catalog_alignment_trace(primary, case)
    assert_warm_add(primary, case)
    reports.append(_case_report(case, primary))

    locations = client.chat(
        message=queries.locations_followup,
        conversation_id=primary_conversation,
    )
    case = "multi_entity_locations"
    assert_success(locations, case)
    assert_exact_edges(locations, association_edges | headquarters_edges, case)
    assert_exact_nodes(locations, location_session_nodes, case)
    _require(
        len(dataset.location_nodes_for_edges(headquarters_edges)) == 9,
        f"{case}: expected exactly 9 unique locations for 10 companies",
    )
    assert_raw_provenance(locations, dataset, case)
    assert_fresh_trace(locations, case, required_tools={"relations"})
    actual_headquarters = {
        edge
        for edge in edge_signatures(locations)
        if edge.raw_relation == "Headquartered_in"
        and edge.source in association_company_ids
    }
    _require(
        actual_headquarters == headquarters_edges,
        f"{case}: incomplete or unsupported location scope; "
        f"missing={sorted(headquarters_edges - actual_headquarters)}, "
        f"unexpected={sorted(actual_headquarters - headquarters_edges)}",
    )
    assert_warm_add(locations, case)
    reports.append(_case_report(case, locations))

    repeated = client.chat(
        message=queries.multi_entity,
        conversation_id=primary_conversation,
    )
    case = "multi_entity_raw_repeat"
    assert_success(repeated, case)
    assert_exact_edges(repeated, association_edges | headquarters_edges, case)
    assert_exact_nodes(repeated, location_session_nodes, case)
    assert_raw_provenance(repeated, dataset, case)
    assert_raw_cache_hit(repeated, case)
    graph_id = str(repeated.get("graph_id"))
    by_conversation = client.get(
        "/graph?" + urllib.parse.urlencode({"conversation_id": primary_conversation})
    )
    by_graph_id = client.get(
        "/graph?" + urllib.parse.urlencode({"graph_id": graph_id})
    )
    _require(
        by_conversation == repeated.get("graph") == by_graph_id,
        f"{case}: /graph selectors do not return the exact persisted session graph",
    )
    reports.append(_case_report(case, repeated))

    paraphrased_multi = client.chat(
        message=queries.multi_entity_paraphrase,
        conversation_id=_new_conversation(),
    )
    case = "multi_entity_paraphrase"
    assert_success(paraphrased_multi, case)
    assert_exact_edges(paraphrased_multi, association_edges, case)
    assert_exact_nodes(paraphrased_multi, association_nodes, case)
    assert_raw_provenance(paraphrased_multi, dataset, case)
    assert_canonical_or_fresh_equivalent_trace(
        paraphrased_multi,
        case,
        required_tools={"companies", "persons", "relations"},
        expected_touch_operation="touch",
    )
    assert_planner_result_merge(paraphrased_multi, "union", case)
    assert_catalog_alignment_trace(paraphrased_multi, case)
    reports.append(_case_report(case, paraphrased_multi))

    ma_yun_founded_edges = dataset.relation_edges(
        subject_ids={"person:P004"},
        raw_relation_types={"Founder_of", "Co-founder_of"},
        direction="outgoing",
    )
    founded = client.chat(
        message=queries.ma_yun_founded,
        conversation_id=_new_conversation(),
    )
    case = "ma_yun_founded"
    assert_success(founded, case)
    assert_exact_edges(founded, ma_yun_founded_edges, case)
    assert_exact_nodes(
        founded,
        {
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        case,
    )
    assert_raw_provenance(founded, dataset, case)
    assert_fresh_trace(founded, case, required_tools={"persons", "relations"})
    founded_scopes = researcher_relation_scopes(founded, case)
    _require(any("founded" in scope for scope in founded_scopes), f"{case}: founded scope missing")
    _require(
        all("controls" not in scope and "owns" not in scope for scope in founded_scopes),
        f"{case}: founded query leaked control/ownership scope",
    )
    assert_warm_add(founded, case)
    reports.append(_case_report(case, founded))

    founded_paraphrase = client.chat(
        message=queries.ma_yun_founded_paraphrase,
        conversation_id=_new_conversation(),
    )
    case = "ma_yun_founded_paraphrase"
    assert_success(founded_paraphrase, case)
    assert_exact_edges(founded_paraphrase, ma_yun_founded_edges, case)
    assert_exact_nodes(
        founded_paraphrase,
        {
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        case,
    )
    assert_raw_provenance(founded_paraphrase, dataset, case)
    assert_canonical_or_fresh_equivalent_trace(
        founded_paraphrase,
        case,
        required_tools={"persons", "relations"},
        expected_touch_operation="promote",
    )
    reports.append(_case_report(case, founded_paraphrase))

    control_edges = dataset.relation_edges(
        subject_ids={"person:P004"},
        raw_relation_types=CONTROL_FALLBACK_RELATIONS,
        direction="outgoing",
    )
    controlled = client.chat(
        message=queries.ma_yun_control,
        conversation_id=_new_conversation(),
    )
    case = "ma_yun_control"
    assert_success(controlled, case)
    assert_exact_edges(controlled, control_edges, case)
    assert_exact_nodes(
        controlled,
        {
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        case,
    )
    assert_raw_provenance(controlled, dataset, case)
    assert_fresh_trace(controlled, case, required_tools={"persons", "relations"})
    assert_control_public_contract(controlled, case)
    assert_warm_add(controlled, case)
    reports.append(
        _case_report(
            case,
            controlled,
            unobservable_constraints=(
                "relations.direction",
                "relations.raw_relation_types",
            ),
        )
    )

    person_owns = client.chat(
        message=queries.ma_yun_owns,
        conversation_id=_new_conversation(),
    )
    case = "ma_yun_owns"
    assert_success(person_owns, case)
    assert_exact_edges(person_owns, set(), case)
    assert_exact_nodes(person_owns, {"person:P004": ("person", "马云")}, case)
    assert_raw_provenance(person_owns, dataset, case)
    assert_fresh_trace(
        person_owns,
        case,
        required_tools={"persons", "relations"},
        route_path="no_results",
    )
    owns_scopes = researcher_relation_scopes(person_owns, case)
    _require(any("owns" in scope for scope in owns_scopes), f"{case}: owns scope missing")
    _require(
        all("controls" not in scope and "founded" not in scope for scope in owns_scopes),
        f"{case}: ownership query leaked control/founder scope",
    )
    assert_no_results_cache_skip(person_owns, case)
    reports.append(_case_report(case, person_owns))

    alibaba_owns_edges = dataset.relation_edges(
        subject_ids={"company:C005"},
        raw_relation_types={"Owns"},
        direction="outgoing",
    )
    alibaba_owns = client.chat(
        message=queries.alibaba_owns,
        conversation_id=_new_conversation(),
    )
    case = "alibaba_owns"
    assert_success(alibaba_owns, case)
    assert_exact_edges(alibaba_owns, alibaba_owns_edges, case)
    assert_exact_nodes(
        alibaba_owns,
        {
            "company:C005": ("company", "阿里巴巴集团"),
            "company:C023": ("company", "阿里云"),
        },
        case,
    )
    assert_raw_provenance(alibaba_owns, dataset, case)
    _require(
        ExpectedEdge("company:C005", "company:C023", "owns", "Owns", 25)
        in edge_signatures(alibaba_owns),
        f"{case}: missing raw C005 Owns C023 row",
    )
    assert_fresh_trace(alibaba_owns, case, required_tools={"companies", "relations"})
    assert_warm_add(alibaba_owns, case)
    reports.append(_case_report(case, alibaba_owns))

    ma_huateng = client.chat(
        message=queries.ma_huateng_founded,
        conversation_id=_new_conversation(),
    )
    case = "ma_huateng_founded"
    assert_success(ma_huateng, case)
    assert_exact_edges(ma_huateng, set(), case)
    assert_exact_nodes(ma_huateng, {"person:P005": ("person", "马化腾")}, case)
    assert_raw_provenance(ma_huateng, dataset, case)
    assert_fresh_trace(
        ma_huateng,
        case,
        required_tools={"persons", "relations"},
        route_path="no_results",
    )
    no_result_scopes = researcher_relation_scopes(ma_huateng, case)
    _require(any("founded" in scope for scope in no_result_scopes), f"{case}: founded scope missing")
    assert_no_results_cache_skip(ma_huateng, case)
    reports.append(_case_report(case, ma_huateng))

    return reports


def _valid_namespace(value: str) -> bool:
    return (
        3 <= len(value) <= 512
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]", value) is not None
        and ".." not in value
    )


def generated_namespace() -> str:
    return f"live_audit_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}_{uuid4().hex[:12]}"


def wait_until_ready(client: ApiClient, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ready = client.get("/ready")
            if ready.get("status") == "ready" and all(
                value is True for value in (ready.get("checks") or {}).values()
            ):
                health = client.get("/health")
                _require(health.get("status") == "ok", "/health did not return ok")
                return
        except AuditFailure as exc:
            last_error = exc
        time.sleep(1)
    raise AuditFailure(f"Docker API was not ready within {timeout_seconds}s: {last_error}")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--namespace", default=generated_namespace())
    parser.add_argument("--skip-build", action="store_true", help="reuse the existing Docker images")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for /ready after Compose recreation",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=615.0,
        help="per-request HTTP timeout; must remain outside the backend 570s bound",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate raw fixtures and print the generated query plan without Docker/API calls",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path (answers are excluded)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        dataset = RawDataset.load(args.data_directory.resolve())
        queries = build_queries(dataset)
        if not _valid_namespace(args.namespace):
            raise AuditFailure(f"invalid Chroma collection namespace: {args.namespace!r}")
        if args.plan_only:
            print(
                _safe_json(
                    {
                        "mode": "plan-only",
                        "namespace": args.namespace,
                        "raw_counts": {
                            "persons": len(dataset.persons),
                            "companies": len(dataset.companies),
                            "relations": len(dataset.relations),
                        },
                        "queries": [
                            {"case": case, "message": message}
                            for case, message in queries.items()
                        ],
                    },
                )
            )
            return 0

        stack = DockerStack(namespace=args.namespace, build=not args.skip_build)
        stack.start()
        client = ApiClient(args.api_base, timeout_seconds=args.request_timeout)
        wait_until_ready(client, args.startup_timeout)
        reports = run_live_audit(client, dataset)

        repeat_request_id = next(
            str(item["request_id"])
            for item in reports
            if item["case"] == "multi_entity_raw_repeat"
        )
        logs = stack.logs()
        _require(repeat_request_id in logs, "backend logs lack the raw-repeat request ID")
        _require("cache_hit" in logs, "backend logs lack a cache_hit event")

        report = {
            "status": "passed",
            "namespace": args.namespace,
            "api_base": _redact_sensitive(args.api_base),
            "cases": reports,
        }
        rendered = _safe_json(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except AuditFailure as exc:
        print(_redact_sensitive(f"live audit failed: {exc}"), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(
            _redact_sensitive(
                f"live audit failed: unexpected {type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
