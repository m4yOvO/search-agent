#!/usr/bin/env python3
"""Explicit, paid outside-in audit generated from all three raw JSON arrays.

The script never imports backend application code and never reads an API key.  It
talks only to the public Docker HTTP API.  Running it without ``--execute`` prints
the bounded plan and performs no network request, which keeps pytest and accidental
shell invocations free of paid model calls.

Examples::

    python scripts/full_dataset_audit.py
    python scripts/full_dataset_audit.py --execute --concurrency 2
    python scripts/full_dataset_audit.py --execute --max-persons 3 \
        --max-companies 3 --max-locations 3 --max-pairs 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_REPORT = PROJECT_ROOT / "output" / "full-dataset-audit.json"
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
ROLE_RELATIONS = frozenset(
    {
        "CEO_of",
        "Chairman_of",
        "Chairwoman_of",
        "Former_CEO_of",
        "Former_Chairman_of",
        "Former_President_of",
        "Founder_of",
        "Co-founder_of",
    }
)
SENSITIVE_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTHORIZATION",
)


class AuditFailure(AssertionError):
    """A bounded audit failure safe to include in the local report."""


@dataclass(frozen=True, order=True)
class ExpectedNode:
    entity_id: str
    entity_type: str
    label: str
    source_file: str


@dataclass(frozen=True, order=True)
class ExpectedEdge:
    record_id: str
    source: str
    target: str
    relation_type: str
    raw_relation: str
    raw_head: str
    raw_tail: str
    source_row: int

    @property
    def evidence_id(self) -> str:
        return f"evidence:raw:relation:{self.source_row:04d}"


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    suite: str
    message: str
    expected_nodes: tuple[ExpectedNode, ...]
    expected_edges: tuple[ExpectedEdge, ...]
    required_tools: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.expected_edges


@dataclass(frozen=True)
class RawDataset:
    persons: tuple[dict[str, Any], ...]
    companies: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    nodes: Mapping[str, ExpectedNode]
    edges: tuple[ExpectedEdge, ...]
    token_to_id: Mapping[str, str]
    base_entity_ids: frozenset[str]

    @classmethod
    def load(cls, directory: Path) -> RawDataset:
        actual_json = {path.name for path in directory.glob("*.json")}
        if actual_json != set(RAW_FILES):
            raise AuditFailure(
                f"data_json_set_mismatch:{','.join(sorted(actual_json))}"
            )
        persons = tuple(_load_array(directory / RAW_FILES[0]))
        companies = tuple(_load_array(directory / RAW_FILES[1]))
        relations = tuple(_load_array(directory / RAW_FILES[2]))
        if (len(persons), len(companies), len(relations)) != (20, 30, 109):
            raise AuditFailure(
                "raw_count_mismatch:"
                f"{len(persons)},{len(companies)},{len(relations)}"
            )
        _validate_schema(persons, {"id", "name", "nationality", "summary"}, "person")
        _validate_schema(
            companies,
            {"id", "name", "legal_rep_id", "city", "founded_year"},
            "company",
        )
        _validate_schema(relations, {"head", "relation", "tail"}, "relation")

        token_to_id: dict[str, str] = {}
        nodes: dict[str, ExpectedNode] = {}
        base_ids: set[str] = set()
        for entity_type, rows, source_file in (
            ("person", persons, "person 1.json"),
            ("company", companies, "company 1.json"),
        ):
            for row in rows:
                raw_id = str(row["id"])
                raw_name = str(row["name"])
                stable_id = f"{entity_type}:{raw_id}"
                node = ExpectedNode(stable_id, entity_type, raw_name, source_file)
                nodes[stable_id] = node
                base_ids.add(stable_id)
                for token in (raw_id, raw_name):
                    previous = token_to_id.setdefault(token, stable_id)
                    if previous != stable_id:
                        raise AuditFailure(f"ambiguous_raw_token:{token}")

        for row in companies:
            city = str(row["city"])
            location_id = f"location:{_slugify(city)}"
            nodes.setdefault(
                location_id,
                ExpectedNode(location_id, "location", city, "company 1.json"),
            )

        def endpoint_id(
            raw_value: str,
            raw_relation: str,
            position: str,
            row_number: int,
        ) -> str:
            known = token_to_id.get(raw_value)
            if known is not None:
                return known
            if raw_relation == "Headquartered_in" and position == "tail":
                entity_id = f"location:{_slugify(raw_value)}"
                nodes.setdefault(
                    entity_id,
                    ExpectedNode(entity_id, "location", raw_value, "relations 1.json"),
                )
                return entity_id
            person_side = raw_relation in ROLE_RELATIONS and position == "head"
            entity_type = "person" if person_side else "company"
            entity_id = f"{entity_type}:raw-reference:{_slugify(raw_value)}"
            nodes.setdefault(
                entity_id,
                ExpectedNode(entity_id, entity_type, raw_value, "relations 1.json"),
            )
            return entity_id

        edges: list[ExpectedEdge] = []
        for row_number, row in enumerate(relations, start=1):
            raw_relation = str(row["relation"])
            relation_type = RAW_TO_TYPED_RELATION.get(raw_relation)
            if relation_type is None:
                raise AuditFailure(f"unknown_raw_relation:{raw_relation}")
            raw_head = str(row["head"])
            raw_tail = str(row["tail"])
            edges.append(
                ExpectedEdge(
                    record_id=f"relation:raw:{row_number:04d}",
                    source=endpoint_id(raw_head, raw_relation, "head", row_number),
                    target=endpoint_id(raw_tail, raw_relation, "tail", row_number),
                    relation_type=relation_type,
                    raw_relation=raw_relation,
                    raw_head=raw_head,
                    raw_tail=raw_tail,
                    source_row=row_number,
                )
            )
        return cls(
            persons=persons,
            companies=companies,
            relations=relations,
            nodes=nodes,
            edges=tuple(edges),
            token_to_id=token_to_id,
            base_entity_ids=frozenset(base_ids),
        )

    def build_cases(
        self,
        *,
        max_persons: int | None = None,
        max_companies: int | None = None,
        max_locations: int | None = None,
        max_pairs: int | None = None,
        skip_empty: bool = False,
    ) -> tuple[AuditCase, ...]:
        people = self._person_cases()[:max_persons]
        companies = self._company_cases()[:max_companies]
        locations = self._location_cases()[:max_locations]
        pairs = self._pair_cases()[:max_pairs]
        cases = (*people, *companies, *locations, *pairs)
        if skip_empty:
            cases = tuple(case for case in cases if not case.is_empty)
        return tuple(cases)

    def _case_nodes(
        self,
        seed_ids: Iterable[str],
        edges: Iterable[ExpectedEdge],
    ) -> tuple[ExpectedNode, ...]:
        ids = set(seed_ids)
        for edge in edges:
            ids.update((edge.source, edge.target))
        return tuple(sorted(self.nodes[entity_id] for entity_id in ids))

    def _person_cases(self) -> tuple[AuditCase, ...]:
        output: list[AuditCase] = []
        for row in self.persons:
            raw_id = str(row["id"])
            name = str(row["name"])
            subject = f"person:{raw_id}"
            edges = tuple(
                edge
                for edge in self.edges
                if edge.raw_relation in BUSINESS_RELATIONS
                and subject in {edge.source, edge.target}
                and any(
                    endpoint.startswith("company:")
                    for endpoint in (edge.source, edge.target)
                )
            )
            output.append(
                AuditCase(
                    case_id=f"person-{raw_id}",
                    suite="persons",
                    message=f"{name}有哪些公司？",
                    expected_nodes=self._case_nodes((subject,), edges),
                    expected_edges=edges,
                    required_tools=("persons", "relations"),
                )
            )
        return tuple(output)

    def _company_cases(self) -> tuple[AuditCase, ...]:
        output: list[AuditCase] = []
        for row in self.companies:
            raw_id = str(row["id"])
            name = str(row["name"])
            subject = f"company:{raw_id}"
            edges = tuple(
                edge
                for edge in self.edges
                if edge.raw_relation in BUSINESS_RELATIONS
                and subject in {edge.source, edge.target}
                and edge.source.startswith("company:")
                and edge.target.startswith("company:")
            )
            output.append(
                AuditCase(
                    case_id=f"company-{raw_id}",
                    suite="companies",
                    message=f"{name}有哪些关联公司？",
                    expected_nodes=self._case_nodes((subject,), edges),
                    expected_edges=edges,
                    required_tools=("companies", "relations"),
                )
            )
        return tuple(output)

    def _location_cases(self) -> tuple[AuditCase, ...]:
        output: list[AuditCase] = []
        for row in self.companies:
            raw_id = str(row["id"])
            name = str(row["name"])
            subject = f"company:{raw_id}"
            edges = tuple(
                edge
                for edge in self.edges
                if edge.source == subject
                and edge.raw_relation == "Headquartered_in"
            )
            if len(edges) != 1:
                raise AuditFailure(f"company_headquarters_count:{raw_id}:{len(edges)}")
            output.append(
                AuditCase(
                    case_id=f"location-{raw_id}",
                    suite="locations",
                    message=f"{name}在哪里？",
                    expected_nodes=self._case_nodes((subject,), edges),
                    expected_edges=edges,
                    required_tools=("companies", "relations"),
                )
            )
        return tuple(output)

    def _pair_cases(self) -> tuple[AuditCase, ...]:
        groups: defaultdict[frozenset[str], list[ExpectedEdge]] = defaultdict(list)
        for edge in self.edges:
            if edge.raw_relation not in BUSINESS_RELATIONS:
                continue
            if (
                edge.source not in self.base_entity_ids
                or edge.target not in self.base_entity_ids
            ):
                continue
            groups[frozenset((edge.source, edge.target))].append(edge)

        output: list[AuditCase] = []
        for number, (pair, rows) in enumerate(
            sorted(groups.items(), key=lambda item: tuple(sorted(item[0]))), start=1
        ):
            ids = tuple(sorted(pair))
            labels = [self.nodes[entity_id].label for entity_id in ids]
            if len(labels) == 1:
                message = f"{labels[0]}与其自身之间有什么关系？"
            else:
                message = f"{labels[0]}与{labels[1]}之间有什么关系？"
            required = {"relations"}
            for entity_id in ids:
                required.add("persons" if entity_id.startswith("person:") else "companies")
            edges = tuple(sorted(rows))
            output.append(
                AuditCase(
                    case_id=f"pair-{number:03d}",
                    suite="pairs",
                    message=message,
                    expected_nodes=self._case_nodes(ids, edges),
                    expected_edges=edges,
                    required_tools=tuple(sorted(required)),
                )
            )
        return tuple(output)


class ApiClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = _validate_base_url(base_url)
        self.timeout_seconds = timeout_seconds

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def chat(self, case: AuditCase) -> dict[str, Any]:
        return self._request(
            "POST",
            "/chat",
            {
                "conversation_id": str(uuid4()),
                "message": case.message,
                "locale": "zh-CN",
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AuditFailure(f"http_status:{exc.code}") from None
        except urllib.error.URLError as exc:
            reason = type(getattr(exc, "reason", exc)).__name__
            raise AuditFailure(f"connection_error:{reason}") from None
        except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuditFailure(f"invalid_http_response:{type(exc).__name__}") from None
        if not isinstance(value, dict):
            raise AuditFailure("http_response_not_object")
        return value


def validate_response(case: AuditCase, body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only structured facts/trace; model-authored answer text is ignored."""

    if body.get("status") != "success" or body.get("error_code") not in {None, ""}:
        raise AuditFailure(f"response_status:{body.get('status')}:{body.get('error_code')}")
    graph = body.get("graph")
    if not isinstance(graph, dict):
        raise AuditFailure("missing_graph")
    nodes = _unique_records(graph.get("nodes"), "node")
    edges = _unique_records(graph.get("edges"), "edge")
    evidence = _unique_records(graph.get("evidence"), "evidence")

    expected_nodes = {node.entity_id: node for node in case.expected_nodes}
    expected_edges = {edge.record_id: edge for edge in case.expected_edges}
    if set(nodes) != set(expected_nodes):
        raise AuditFailure(
            f"node_id_mismatch:missing={_joined(set(expected_nodes)-set(nodes))}:"
            f"unexpected={_joined(set(nodes)-set(expected_nodes))}"
        )
    if set(edges) != set(expected_edges):
        raise AuditFailure(
            f"edge_id_mismatch:missing={_joined(set(expected_edges)-set(edges))}:"
            f"unexpected={_joined(set(edges)-set(expected_edges))}"
        )

    for entity_id, expected in expected_nodes.items():
        node = nodes[entity_id]
        if (node.get("type"), node.get("label")) != (
            expected.entity_type,
            expected.label,
        ):
            raise AuditFailure(f"node_projection_mismatch:{entity_id}")
        properties = node.get("properties")
        if not isinstance(properties, dict) or properties.get("source_file") != expected.source_file:
            raise AuditFailure(f"node_provenance_mismatch:{entity_id}")

    for record_id, expected in expected_edges.items():
        edge = edges[record_id]
        properties = edge.get("properties")
        if not isinstance(properties, dict):
            raise AuditFailure(f"edge_properties_missing:{record_id}")
        actual = (
            edge.get("source"),
            edge.get("target"),
            edge.get("type"),
            edge.get("label"),
            properties.get("raw_head"),
            properties.get("raw_relation"),
            properties.get("raw_tail"),
            properties.get("source_file"),
            properties.get("source_row"),
        )
        wanted = (
            expected.source,
            expected.target,
            expected.relation_type,
            expected.raw_relation,
            expected.raw_head,
            expected.raw_relation,
            expected.raw_tail,
            "relations 1.json",
            expected.source_row,
        )
        if actual != wanted:
            raise AuditFailure(f"edge_provenance_mismatch:{record_id}")
        evidence_item = evidence.get(expected.evidence_id)
        if evidence_item is None:
            raise AuditFailure(f"relation_evidence_missing:{record_id}")
        if (
            evidence_item.get("record_id") != f"relations 1.json#{expected.source_row}"
            or evidence_item.get("source_kind") != "raw_relation"
        ):
            raise AuditFailure(f"relation_evidence_mismatch:{record_id}")

    referenced_evidence = {
        str(evidence_id)
        for record in (*nodes.values(), *edges.values())
        for evidence_id in _string_list(record.get("evidence_ids"))
    }
    if not referenced_evidence or referenced_evidence != set(evidence):
        raise AuditFailure("evidence_catalog_mismatch")
    if any(
        item.get("provider") != "local-raw-json-mock"
        or item.get("is_demo") is not True
        for item in evidence.values()
    ):
        raise AuditFailure("evidence_fact_boundary_mismatch")

    trace = body.get("trace")
    if not isinstance(trace, dict):
        raise AuditFailure("missing_trace")
    counters = {
        key: _nonnegative_int(trace.get(key), key)
        for key in (
            "model_calls",
            "planner_model_calls",
            "researcher_model_calls",
            "visualizer_model_calls",
            "tool_calls",
            "research_steps",
            "replans",
        )
    }
    if counters["model_calls"] != sum(
        counters[key]
        for key in (
            "planner_model_calls",
            "researcher_model_calls",
            "visualizer_model_calls",
        )
    ):
        raise AuditFailure("model_counter_mismatch")
    if not isinstance(trace.get("route_history"), list):
        raise AuditFailure("missing_route_history")

    memory = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    cache_hit = memory.get("cache_hit") is True
    if cache_hit:
        if counters["model_calls"] != 0 or counters["tool_calls"] != 0:
            raise AuditFailure("cache_hit_executed_agent_work")
    else:
        if (
            trace.get("researcher_invoked") is not True
            or counters["planner_model_calls"] < 1
            or counters["researcher_model_calls"] < 1
            or counters["visualizer_model_calls"] < 1
            or counters["tool_calls"] < 1
        ):
            raise AuditFailure("fresh_trace_missing_agent_work")
        steps = trace.get("agent_steps")
        if not isinstance(steps, list):
            raise AuditFailure("missing_agent_steps")
        called_tools = {
            str(step.get("tool"))
            for step in steps
            if isinstance(step, dict)
            and step.get("role") == "researcher"
            and step.get("action") == "tool_result"
            and step.get("tool")
        }
        missing_tools = set(case.required_tools) - called_tools
        if missing_tools:
            raise AuditFailure(f"required_tool_missing:{_joined(missing_tools)}")

    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "passed": True,
        "expected_nodes": len(expected_nodes),
        "expected_edges": len(expected_edges),
        "actual_nodes": len(nodes),
        "actual_edges": len(edges),
        "cache_hit": cache_hit,
        "cache_match_type": memory.get("match_type"),
        **counters,
    }


def execute_plan(
    *,
    client: ApiClient,
    cases: Sequence[AuditCase],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run independent-conversation cases concurrently and preserve plan order."""

    results: list[dict[str, Any] | None] = [None] * len(cases)

    def run_one(index: int, case: AuditCase) -> tuple[int, dict[str, Any]]:
        try:
            body = client.chat(case)
            return index, validate_response(case, body)
        except Exception as exc:  # every case should leave a bounded report row
            error = _redact(str(exc))[:500] or type(exc).__name__
            return index, {
                "case_id": case.case_id,
                "suite": case.suite,
                "passed": False,
                "error": error,
                "expected_nodes": len(case.expected_nodes),
                "expected_edges": len(case.expected_edges),
            }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: list[Future[tuple[int, dict[str, Any]]]] = [
            executor.submit(run_one, index, case)
            for index, case in enumerate(cases)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            outcome = "PASS" if result["passed"] else "FAIL"
            print(f"{outcome} {result['case_id']}", file=sys.stderr, flush=True)
    return [result for result in results if result is not None]


def plan_summary(cases: Sequence[AuditCase], dataset: RawDataset) -> dict[str, Any]:
    counts = Counter(case.suite for case in cases)
    return {
        "mode": "plan-only",
        "network_requests": 0,
        "raw_counts": {
            "persons": len(dataset.persons),
            "companies": len(dataset.companies),
            "relations": len(dataset.relations),
        },
        "case_count": len(cases),
        "suite_counts": dict(sorted(counts.items())),
        "empty_expected_case_count": sum(case.is_empty for case in cases),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly allow paid HTTP /chat requests. Omit for a zero-network plan.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--timeout", type=_positive_float, default=615.0)
    parser.add_argument("--max-persons", type=_nonnegative_int_argument)
    parser.add_argument("--max-companies", type=_nonnegative_int_argument)
    parser.add_argument("--max-locations", type=_nonnegative_int_argument)
    parser.add_argument("--max-pairs", type=_nonnegative_int_argument)
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip data-derived cases whose expected raw relation set is empty.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency > 16:
        raise SystemExit("--concurrency must not exceed 16")
    dataset = RawDataset.load(args.data_directory.resolve())
    cases = dataset.build_cases(
        max_persons=args.max_persons,
        max_companies=args.max_companies,
        max_locations=args.max_locations,
        max_pairs=args.max_pairs,
        skip_empty=args.skip_empty,
    )
    if not args.execute:
        print(_safe_json(plan_summary(cases, dataset)))
        return 0
    if not cases:
        raise SystemExit("execution plan is empty")

    client = ApiClient(args.base_url, args.timeout)
    for path in ("/health", "/ready"):
        status = client.get(path).get("status")
        if status not in {"ok", "ready", "healthy"}:
            raise AuditFailure(f"service_not_ready:{path}:{status}")

    results = execute_plan(client=client, cases=cases, concurrency=args.concurrency)
    failures = [result for result in results if result.get("passed") is not True]
    report = {
        "mode": "executed",
        "base_url": client.base_url,
        "raw_counts": {
            "persons": len(dataset.persons),
            "companies": len(dataset.companies),
            "relations": len(dataset.relations),
        },
        "case_count": len(cases),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "suite_counts": dict(sorted(Counter(case.suite for case in cases).items())),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_safe_json(report) + "\n", encoding="utf-8")
    print(
        _safe_json(
            {
                "mode": "executed",
                "case_count": len(cases),
                "passed": report["passed"],
                "failed": report["failed"],
                "report": str(args.report.resolve()),
            }
        )
    )
    return 1 if failures else 0


def _load_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"raw_file_unreadable:{path.name}:{type(exc).__name__}") from None
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AuditFailure(f"raw_file_not_object_array:{path.name}")
    return value


def _validate_schema(
    rows: Iterable[Mapping[str, Any]], required: set[str], kind: str
) -> None:
    for row_number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise AuditFailure(f"raw_schema_mismatch:{kind}:{row_number}")


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(
        r"[^\w\u4e00-\u9fff]+", "-", normalized, flags=re.UNICODE
    ).strip("-")
    if not normalized:
        raise AuditFailure("raw_endpoint_empty_slug")
    return normalized


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise AuditFailure("invalid_base_url")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "", "", "")
    ).rstrip("/")


def _unique_records(value: Any, kind: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise AuditFailure(f"invalid_{kind}_list")
    records: dict[str, dict[str, Any]] = {}
    for item in value:
        record_id = item.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise AuditFailure(f"invalid_{kind}_id")
        if record_id in records:
            raise AuditFailure(f"duplicate_{kind}_id:{record_id}")
        records[record_id] = item
    return records


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AuditFailure("invalid_evidence_reference_list")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AuditFailure(f"invalid_trace_counter:{field}")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int_argument(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _joined(values: Iterable[str]) -> str:
    return ",".join(sorted(values)) or "none"


def _redact(value: str, environment: Mapping[str, str] | None = None) -> str:
    redacted = value
    env = os.environ if environment is None else environment
    secrets = sorted(
        {
            str(secret)
            for name, secret in env.items()
            if secret
            and len(str(secret)) >= 4
            and any(marker in name.upper() for marker in SENSITIVE_MARKERS)
        },
        key=len,
        reverse=True,
    )
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", redacted)
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        redacted,
    )
    return redacted


def _safe_json(value: Any) -> str:
    return _redact(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(_safe_json({"status": "failed", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from None
