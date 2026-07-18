from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = ROOT / "data"
SCRIPT_PATH = ROOT / "scripts" / "live_audit.py"
FULL_DATASET_SCRIPT_PATH = ROOT / "scripts" / "full_dataset_audit.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("project_live_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def _load_full_dataset_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "project_full_dataset_audit", FULL_DATASET_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full_audit = _load_full_dataset_script()


def _evidence(evidence_id: str) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "provider": "local-raw-json-mock",
        "record_id": evidence_id.replace("evidence:", ""),
        "source_kind": "raw_relation",
        "updated_at": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "is_demo": True,
        "source_url": None,
    }


def _graph_body(
    *,
    edges: set[Any],
    node_labels: dict[str, tuple[str, str]],
    cache_hit: bool = False,
    match_type: str | None = None,
    cache_status: str | None = None,
    fresh: bool = True,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    for entity_id, (node_type, label) in node_labels.items():
        evidence_id = f"evidence:{entity_id}"
        evidence.append(_evidence(evidence_id))
        nodes.append(
            {
                "id": entity_id,
                "type": node_type,
                "label": label,
                "properties": {},
                "evidence_ids": [evidence_id],
            }
        )
    for expected in sorted(edges):
        evidence_id = f"evidence:relation:{expected.source_row:04d}"
        evidence.append(_evidence(evidence_id))
        graph_edges.append(
            {
                "id": f"relation:raw:{expected.source_row:04d}",
                "source": expected.source,
                "target": expected.target,
                "type": expected.relation_type,
                "label": expected.raw_relation,
                "properties": {
                    "raw_relation": expected.raw_relation,
                    "source_row": expected.source_row,
                },
                "evidence_ids": [evidence_id],
            }
        )

    if fresh:
        trace = {
            "researcher_invoked": True,
            "tool_calls": 2,
            "research_steps": 3,
            "replans": 0,
            "model_provider": "openai",
            "model_name": "test-openai-model",
            "model_calls": 5,
            "planner_model_calls": 1,
            "researcher_model_calls": 3,
            "visualizer_model_calls": 1,
            "prompt_versions": {},
            "route_history": [
                "begin_turn",
                "raw_cache_probe",
                "planner",
                "researcher",
                "result_gate",
                "researcher",
                "result_gate",
                "researcher",
                "result_gate",
                "canonical_cache_probe",
                "visualizer",
                "memory_write",
                "merge_session_graph",
                "compact_session",
            ],
            "agent_steps": [
                {
                    "role": "planner",
                    "action": "plan",
                    "tool": None,
                    "relation_types": [],
                    "result_merge": "not_applicable",
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [],
                    "error_code": None,
                },
                {
                    "role": "researcher",
                    "action": "call_tool",
                    "tool": "persons",
                    "relation_types": [],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [],
                    "error_code": None,
                },
                {
                    "role": "researcher",
                    "action": "tool_result",
                    "tool": "persons",
                    "relation_types": [],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": ["person:P004"],
                    "error_code": None,
                },
                {
                    "role": "researcher",
                    "action": "call_tool",
                    "tool": "relations",
                    "relation_types": ["founded"],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [],
                    "error_code": None,
                },
                {
                    "role": "researcher",
                    "action": "tool_result",
                    "tool": "relations",
                    "relation_types": ["founded"],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [edge_id["id"] for edge_id in graph_edges],
                    "error_code": None,
                },
                {
                    "role": "researcher",
                    "action": "finish",
                    "tool": None,
                    "relation_types": ["founded"],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [edge_id["id"] for edge_id in graph_edges],
                    "error_code": None,
                },
                {
                    "role": "visualizer",
                    "action": "select_records",
                    "tool": None,
                    "relation_types": [],
                    "result_merge": None,
                    "resolution_strategy": None,
                    "resolution_version": None,
                    "record_ids": [edge_id["id"] for edge_id in graph_edges],
                    "error_code": None,
                },
            ],
        }
    else:
        trace = {
            "researcher_invoked": False,
            "tool_calls": 0,
            "research_steps": 0,
            "replans": 0,
            "model_provider": "openai",
            "model_name": "test-openai-model",
            "model_calls": 0,
            "planner_model_calls": 0,
            "researcher_model_calls": 0,
            "visualizer_model_calls": 0,
            "prompt_versions": {},
            "route_history": [
                "begin_turn",
                "raw_cache_probe",
                "cache_hydrate",
                "cache_touch",
                "merge_session_graph",
                "compact_session",
            ],
            "agent_steps": [],
        }

    return {
        "conversation_id": "00000000-0000-4000-8000-000000000001",
        "request_id": "request:test",
        "status": "success",
        "error_code": None,
        "answer": f"free-form answer intentionally ignored by the audit {audit.ZH_DEMO_DISCLAIMER}",
        "graph_id": "graph:test",
        "graph": {
            "graph_id": "graph:test",
            "nodes": nodes,
            "edges": graph_edges,
            "evidence": evidence,
            "generated_at": "2026-01-01T00:00:00Z",
            "data_version": "raw-v1-test",
        },
        "memory": {
            "cache_hit": cache_hit,
            "match_type": match_type,
            "status": cache_status,
            "write_operation": "promote" if cache_hit else "add",
            "result_id": "cache:test" if cache_hit else "cache:fresh",
        },
        "trace": trace,
        "disclaimer": audit.ZH_DEMO_DISCLAIMER,
    }


def test_live_audit_plan_is_derived_from_the_three_raw_fixtures() -> None:
    dataset = audit.RawDataset.load(DATA_DIRECTORY)
    queries = audit.build_queries(dataset)

    assert (len(dataset.persons), len(dataset.companies), len(dataset.relations)) == (
        20,
        30,
        109,
    )
    assert queries.multi_entity == "特斯拉和马斯克有哪些关联公司？"
    assert queries.locations_followup == "这些公司在哪？"
    assert queries.multi_entity_paraphrase != queries.multi_entity
    assert {case for case, _ in queries.items()} == {
        "catalog_aligned_multi_entity",
        "multi_entity_locations",
        "multi_entity_raw_repeat",
        "multi_entity_paraphrase",
        "ma_yun_founded",
        "ma_yun_founded_paraphrase",
        "ma_yun_control",
        "ma_yun_owns",
        "alibaba_owns",
        "ma_huateng_founded",
    }


def test_fixture_expectations_cover_multi_entity_duplicates_locations_and_semantics() -> None:
    dataset = audit.RawDataset.load(DATA_DIRECTORY)
    association = dataset.relation_edges(
        subject_ids={"person:P001", "company:C001"},
        raw_relation_types=audit.BUSINESS_RELATIONS,
        direction="any",
    )

    assert len(association) == 14
    assert {
        (edge.source, edge.target, edge.raw_relation, edge.source_row)
        for edge in association
    } >= {
        ("person:P001", "company:C001", "CEO_of", 1),
        ("person:P001", "company:C001", "CEO_of", 105),
        ("person:P001", "company:C002", "Founder_of", 2),
        ("person:P001", "company:C021", "Founder_of", 3),
        ("company:C001", "company:C025", "Partner_with", 35),
        ("company:C001", "company:C025", "Partner_with", 107),
        ("company:C025", "company:C001", "Supplier_to", 42),
        ("company:C027", "company:C001", "Competes_with", 57),
        ("company:C001", "company:C030", "Uses_AI_from", 101),
    }
    assert all(edge.raw_relation != "Headquartered_in" for edge in association)

    company_ids = {
        endpoint
        for edge in association
        for endpoint in (edge.source, edge.target)
        if endpoint.startswith("company:")
    }
    # Union is calculated per subject, so Tesla is Musk's opposite company
    # endpoint even though Tesla is also the second explicit subject.
    followup_company_ids = company_ids
    locations = dataset.headquarters_edges(followup_company_ids)
    assert len(company_ids) == len(followup_company_ids) == len(locations) == 10
    assert {
        (edge.source, edge.target, edge.source_row) for edge in locations
    } >= {
        ("company:C001", "location:austin", 60),
        ("company:C002", "location:hawthorne", 61),
        ("company:C021", "location:san-francisco", 80),
    }

    founded = dataset.relation_edges(
        subject_ids={"person:P004"},
        raw_relation_types={"Founder_of", "Co-founder_of"},
        direction="outgoing",
    )
    assert {edge.source_row for edge in founded} == {6, 106}
    assert {edge.raw_relation for edge in founded} == {"Founder_of"}

    alibaba_owns = dataset.relation_edges(
        subject_ids={"company:C005"},
        raw_relation_types={"Owns"},
        direction="outgoing",
    )
    assert alibaba_owns == {
        audit.ExpectedEdge(
            "company:C005", "company:C023", "owns", "Owns", 25
        )
    }

    ma_huateng_founded = dataset.relation_edges(
        subject_ids={"person:P005"},
        raw_relation_types={"Founder_of", "Co-founder_of"},
        direction="outgoing",
    )
    assert ma_huateng_founded == set()


def test_response_assertions_use_raw_graph_and_trace_not_answer_wording() -> None:
    expected = {
        audit.ExpectedEdge(
            "person:P004", "company:C005", "founded", "Founder_of", 6
        )
    }
    body = _graph_body(
        edges=expected,
        node_labels={
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        fresh=True,
    )

    audit.assert_success(body, "fixture_case")
    audit.assert_exact_edges(body, expected, "fixture_case")
    audit.assert_exact_nodes(
        body,
        {
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        "fixture_case",
    )
    audit.assert_fresh_trace(
        body,
        "fixture_case",
        required_tools={"persons", "relations"},
    )

    # Arbitrary prose changes do not affect the audit.
    body["answer"] = f"模型可自由改写这段回答。 {audit.ZH_DEMO_DISCLAIMER}"
    audit.assert_success(body, "fixture_case")

    body["graph"]["edges"][0]["properties"]["raw_relation"] = "CEO_of"
    with pytest.raises(audit.AuditFailure, match="edge mismatch"):
        audit.assert_exact_edges(body, expected, "fixture_case")


def test_raw_cache_assertion_requires_zero_agent_model_and_tool_work() -> None:
    body = _graph_body(
        edges=set(),
        node_labels={},
        cache_hit=True,
        match_type="raw_exact",
        cache_status="hot",
        fresh=False,
    )
    audit.assert_success(body, "cache_repeat")
    audit.assert_raw_cache_hit(body, "cache_repeat")

    body["trace"]["researcher_model_calls"] = 1
    body["trace"]["model_calls"] = 1
    with pytest.raises(audit.AuditFailure, match="non-zero model_calls"):
        audit.assert_raw_cache_hit(body, "cache_repeat")


def test_plan_only_mode_requires_no_key_docker_or_api(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = audit.main(
        [
            "--plan-only",
            "--namespace",
            "unit_test_live_audit_001",
            "--data-directory",
            str(DATA_DIRECTORY),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "plan-only"
    assert output["raw_counts"] == {
        "persons": 20,
        "companies": 30,
        "relations": 109,
    }
    assert len(output["queries"]) == 10


def test_catalog_alignment_and_planner_trace_are_exact_and_role_scoped() -> None:
    body = _graph_body(edges=set(), node_labels={}, fresh=True)
    steps = body["trace"]["agent_steps"]
    planner = next(step for step in steps if step["role"] == "planner")
    planner["result_merge"] = "union"
    person_result = next(
        step
        for step in steps
        if step["role"] == "researcher"
        and step["action"] == "tool_result"
        and step["tool"] == "persons"
    )
    person_result.update(
        {
            "resolution_strategy": "exact",
            "resolution_version": "entity-match-v1",
            "record_ids": ["person:P001"],
        }
    )
    steps.insert(
        -1,
        {
            "role": "researcher",
            "action": "tool_result",
            "tool": "companies",
            "relation_types": [],
            "result_merge": None,
            "resolution_strategy": "exact",
            "resolution_version": "entity-match-v1",
            "record_ids": ["company:C001"],
            "error_code": None,
        },
    )

    audit.assert_planner_result_merge(body, "union", "catalog")
    audit.assert_catalog_alignment_trace(body, "catalog")

    steps[-2]["resolution_version"] = "wrong-version"
    with pytest.raises(audit.AuditFailure, match="catalog-alignment trace mismatch"):
        audit.assert_catalog_alignment_trace(body, "catalog")


def test_graph_list_uniqueness_and_raw_provenance_are_not_hidden_by_sets() -> None:
    dataset = audit.RawDataset.load(DATA_DIRECTORY)
    expected = {
        audit.ExpectedEdge(
            "person:P004", "company:C005", "founded", "Founder_of", 6
        )
    }
    body = _graph_body(
        edges=expected,
        node_labels={
            "person:P004": ("person", "马云"),
            "company:C005": ("company", "阿里巴巴集团"),
        },
        fresh=True,
    )
    body["graph"]["nodes"][0]["properties"]["source_file"] = "person 1.json"
    body["graph"]["nodes"][1]["properties"]["source_file"] = "company 1.json"
    person_evidence = next(
        item
        for item in body["graph"]["evidence"]
        if item["id"] == body["graph"]["nodes"][0]["evidence_ids"][0]
    )
    person_evidence.update({"record_id": "P004", "source_kind": "raw_person"})
    company_evidence = next(
        item
        for item in body["graph"]["evidence"]
        if item["id"] == body["graph"]["nodes"][1]["evidence_ids"][0]
    )
    company_evidence.update({"record_id": "C005", "source_kind": "raw_company"})
    edge = body["graph"]["edges"][0]
    edge["properties"].update(
        {
            "raw_head": "P004",
            "raw_relation": "Founder_of",
            "raw_tail": "C005",
            "source_file": "relations 1.json",
        }
    )
    relation_evidence = next(
        item
        for item in body["graph"]["evidence"]
        if item["id"] == edge["evidence_ids"][0]
    )
    relation_evidence.update(
        {
            "record_id": "relations 1.json#6",
            "source_kind": "raw_relation",
        }
    )

    audit.assert_graph_integrity(body, "provenance")
    audit.assert_raw_provenance(body, dataset, "provenance")

    duplicate_node = copy.deepcopy(body)
    duplicate_node["graph"]["nodes"].append(
        copy.deepcopy(duplicate_node["graph"]["nodes"][0])
    )
    with pytest.raises(audit.AuditFailure, match="duplicate node IDs"):
        audit.assert_graph_integrity(duplicate_node, "duplicate")

    duplicate_edge = copy.deepcopy(body)
    duplicate_edge["graph"]["edges"].append(
        copy.deepcopy(duplicate_edge["graph"]["edges"][0])
    )
    with pytest.raises(audit.AuditFailure, match="duplicate edge IDs"):
        audit.assert_graph_integrity(duplicate_edge, "duplicate")

    duplicate_evidence = copy.deepcopy(body)
    duplicate_evidence["graph"]["evidence"].append(
        copy.deepcopy(duplicate_evidence["graph"]["evidence"][0])
    )
    with pytest.raises(audit.AuditFailure, match="duplicate evidence IDs"):
        audit.assert_graph_integrity(duplicate_evidence, "duplicate")


def test_memory_routes_budget_and_no_results_contracts_are_strict() -> None:
    warm = _graph_body(edges=set(), node_labels={}, fresh=True)
    warm["memory"].update(
        {
            "cache_hit": False,
            "status": "warm",
            "write_operation": "add",
            "result_id": "cache:warm",
        }
    )
    audit.assert_warm_add(warm, "warm")
    audit.assert_fresh_trace(
        warm,
        "warm",
        required_tools={"persons", "relations"},
    )

    no_results = copy.deepcopy(warm)
    researcher_calls = no_results["trace"]["researcher_model_calls"]
    no_results["trace"]["route_history"] = [
        "begin_turn",
        "raw_cache_probe",
        "planner",
        *(["researcher", "result_gate"] * researcher_calls),
        "visualizer",
        "memory_write",
        "merge_session_graph",
        "compact_session",
    ]
    no_results["memory"].update(
        {
            "status": None,
            "write_operation": "skip",
            "result_id": None,
        }
    )
    audit.assert_no_results_cache_skip(no_results, "empty")
    audit.assert_fresh_trace(
        no_results,
        "empty",
        required_tools={"persons", "relations"},
        route_path="no_results",
    )

    exhausted = copy.deepcopy(warm)
    exhausted["trace"]["agent_steps"][1]["error_code"] = "tool_call_limit"
    with pytest.raises(audit.AuditFailure, match="contains Agent error"):
        audit.assert_fresh_trace(
            exhausted,
            "exhausted",
            required_tools={"persons", "relations"},
        )


def test_control_disclosure_and_two_visible_phases_are_ordered() -> None:
    body = _graph_body(edges=set(), node_labels={}, fresh=True)
    body["answer"] = (
        f"模型生成的事实表述。 {audit.ZH_CONTROL_DISCLOSURE} "
        f"{audit.ZH_DEMO_DISCLAIMER}"
    )
    body["trace"]["agent_steps"] = [
        {
            "role": "researcher",
            "action": "call_tool",
            "tool": "relations",
            "relation_types": ["controls"],
        },
        {
            "role": "researcher",
            "action": "tool_result",
            "tool": "relations",
            "relation_types": ["controls"],
            "record_ids": [],
            "count": 0,
        },
        {
            "role": "researcher",
            "action": "call_tool",
            "tool": "relations",
            "relation_types": ["founded", "works_at", "owns"],
        },
        {
            "role": "researcher",
            "action": "tool_result",
            "tool": "relations",
            "relation_types": ["founded", "works_at", "owns"],
            "record_ids": ["relation:raw:0006", "relation:raw:0106"],
            "count": 2,
        },
    ]

    audit.assert_control_public_contract(body, "control")
    body["trace"]["agent_steps"][0]["relation_types"] = ["founded"]
    with pytest.raises(audit.AuditFailure, match="out of order"):
        audit.assert_control_public_contract(body, "control")


def test_http_compose_and_report_errors_redact_sentinel_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sk-sentinel-live-audit-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)

    rendered = audit._safe_json(
        {
            "authorization": f"Bearer {sentinel}",
            "OPENAI_API_KEY": sentinel,
        }
    )
    assert sentinel not in rendered
    assert "[REDACTED]" in rendered

    stack = audit.DockerStack(namespace="redaction_test_001", build=False)

    def compose_failure(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr=f'compose failed with OPENAI_API_KEY="{sentinel}"',
        )

    monkeypatch.setattr(audit.subprocess, "run", compose_failure)
    with pytest.raises(audit.AuditFailure) as compose_error:
        stack._run(["docker", "compose", "config"])
    assert sentinel not in str(compose_error.value)
    assert "[REDACTED]" in str(compose_error.value)

    http_error = urllib.error.HTTPError(
        "http://127.0.0.1:8000/chat",
        500,
        "failure",
        {},
        BytesIO(f'Authorization: Bearer {sentinel}'.encode()),
    )

    def url_failure(*args: Any, **kwargs: Any) -> None:
        raise http_error

    monkeypatch.setattr(audit.urllib.request, "urlopen", url_failure)
    with pytest.raises(audit.AuditFailure) as api_error:
        audit.ApiClient("http://127.0.0.1:8000", 1).get("/health")
    assert sentinel not in str(api_error.value)
    assert "[REDACTED]" in str(api_error.value)


def _full_audit_body(case: Any, *, cache_hit: bool = False) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for expected in case.expected_nodes:
        evidence_id = f"evidence:test:{expected.entity_id}"
        nodes.append(
            {
                "id": expected.entity_id,
                "type": expected.entity_type,
                "label": expected.label,
                "properties": {"source_file": expected.source_file},
                "evidence_ids": [evidence_id],
            }
        )
        evidence.append(
            {
                "id": evidence_id,
                "provider": "local-raw-json-mock",
                "record_id": expected.entity_id,
                "source_kind": f"raw_{expected.entity_type}",
                "is_demo": True,
            }
        )
    for expected in case.expected_edges:
        edges.append(
            {
                "id": expected.record_id,
                "source": expected.source,
                "target": expected.target,
                "type": expected.relation_type,
                "label": expected.raw_relation,
                "properties": {
                    "raw_head": expected.raw_head,
                    "raw_relation": expected.raw_relation,
                    "raw_tail": expected.raw_tail,
                    "source_file": "relations 1.json",
                    "source_row": expected.source_row,
                },
                "evidence_ids": [expected.evidence_id],
            }
        )
        evidence.append(
            {
                "id": expected.evidence_id,
                "provider": "local-raw-json-mock",
                "record_id": f"relations 1.json#{expected.source_row}",
                "source_kind": "raw_relation",
                "is_demo": True,
            }
        )

    if cache_hit:
        trace = {
            "researcher_invoked": False,
            "model_calls": 0,
            "planner_model_calls": 0,
            "researcher_model_calls": 0,
            "visualizer_model_calls": 0,
            "tool_calls": 0,
            "research_steps": 0,
            "replans": 0,
            "route_history": ["raw_cache_probe", "cache_hydrate"],
            "agent_steps": [],
        }
    else:
        researcher_calls = len(case.required_tools) + 1
        trace = {
            "researcher_invoked": True,
            "model_calls": researcher_calls + 2,
            "planner_model_calls": 1,
            "researcher_model_calls": researcher_calls,
            "visualizer_model_calls": 1,
            "tool_calls": len(case.required_tools),
            "research_steps": researcher_calls,
            "replans": 0,
            "route_history": ["planner", "researcher", "visualizer"],
            "agent_steps": [
                {
                    "role": "researcher",
                    "action": "tool_result",
                    "tool": tool,
                }
                for tool in case.required_tools
            ],
        }
    return {
        "status": "success",
        "error_code": None,
        "answer": "模型可以自由改写，full audit 不检查该字段。",
        "graph": {
            "graph_id": "graph:test-full",
            "nodes": nodes,
            "edges": edges,
            "evidence": evidence,
            "data_version": "raw-v1-test",
        },
        "memory": {
            "cache_hit": cache_hit,
            "match_type": "raw_exact" if cache_hit else None,
        },
        "trace": trace,
    }


def test_full_dataset_audit_plan_is_entirely_data_derived() -> None:
    dataset = full_audit.RawDataset.load(DATA_DIRECTORY)
    cases = dataset.build_cases()
    suite_counts: dict[str, int] = {}
    for case in cases:
        suite_counts[case.suite] = suite_counts.get(case.suite, 0) + 1

    assert suite_counts == {
        "persons": 20,
        "companies": 30,
        "locations": 30,
        "pairs": 57,
    }
    assert len(cases) == 137
    assert sum(case.is_empty for case in cases) == 4
    assert len(dataset.edges) == 109

    bounded = dataset.build_cases(
        max_persons=2,
        max_companies=3,
        max_locations=4,
        max_pairs=5,
    )
    assert len(bounded) == 14
    assert [case.suite for case in bounded] == [
        "persons",
        "persons",
        "companies",
        "companies",
        "companies",
        "locations",
        "locations",
        "locations",
        "locations",
        "pairs",
        "pairs",
        "pairs",
        "pairs",
        "pairs",
    ]
    assert len(dataset.build_cases(skip_empty=True)) == 133


def test_full_dataset_audit_default_mode_performs_no_network_or_report_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "must-not-exist.json"
    result = full_audit.main(
        [
            "--data-directory",
            str(DATA_DIRECTORY),
            "--max-persons",
            "1",
            "--max-companies",
            "1",
            "--max-locations",
            "1",
            "--max-pairs",
            "1",
            "--report",
            str(report),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "case_count": 4,
        "empty_expected_case_count": 0,
        "mode": "plan-only",
        "network_requests": 0,
        "raw_counts": {"companies": 30, "persons": 20, "relations": 109},
        "suite_counts": {
            "companies": 1,
            "locations": 1,
            "pairs": 1,
            "persons": 1,
        },
    }
    assert not report.exists()


def test_full_dataset_response_validation_ignores_answer_but_checks_raw_graph() -> None:
    dataset = full_audit.RawDataset.load(DATA_DIRECTORY)
    location_case = next(
        case for case in dataset.build_cases() if case.case_id == "location-C001"
    )
    body = _full_audit_body(location_case)

    result = full_audit.validate_response(location_case, body)
    assert result["passed"] is True
    assert result["actual_edges"] == 1
    assert "answer" not in result
    assert "message" not in result

    body["answer"] = "完全不同的自然语言回答仍不影响结构化事实审计。"
    full_audit.validate_response(location_case, body)

    body["graph"]["edges"][0]["properties"]["raw_tail"] = "wrong-city"
    with pytest.raises(full_audit.AuditFailure, match="edge_provenance_mismatch"):
        full_audit.validate_response(location_case, body)


def test_full_dataset_response_validation_accepts_only_zero_work_cache_hits() -> None:
    dataset = full_audit.RawDataset.load(DATA_DIRECTORY)
    case = dataset.build_cases(max_persons=1, max_companies=0, max_locations=0, max_pairs=0)[0]
    body = _full_audit_body(case, cache_hit=True)

    result = full_audit.validate_response(case, body)
    assert result["cache_hit"] is True
    assert result["model_calls"] == result["tool_calls"] == 0

    body["trace"]["tool_calls"] = 1
    with pytest.raises(full_audit.AuditFailure, match="cache_hit_executed_agent_work"):
        full_audit.validate_response(case, body)


def test_full_dataset_audit_redacts_secrets_and_rejects_credential_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sk-full-dataset-audit-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    rendered = full_audit._safe_json(
        {"error": f"Authorization: Bearer {sentinel}"}
    )
    assert sentinel not in rendered
    assert "[REDACTED]" in rendered

    with pytest.raises(full_audit.AuditFailure, match="invalid_base_url"):
        full_audit.ApiClient(f"http://user:{sentinel}@127.0.0.1:8000", 1)
