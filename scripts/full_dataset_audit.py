#!/usr/bin/env python3
"""Explicit, paid outside-in audit generated from all three raw JSON arrays.

The script never imports backend application code and never reads an API key.  It
talks only to the public Docker HTTP API.  Running it without ``--execute`` prints
the bounded plan and performs no network request, which keeps pytest and accidental
shell invocations free of paid model calls.

Executed runs atomically checkpoint bounded, query-free progress.  A checkpoint is
diagnostic evidence, not a resume manifest: rerunning an interrupted multi-turn audit
must use a fresh cache namespace so conversation and raw-cache semantics are replayed.

Examples::

    python scripts/full_dataset_audit.py
    python scripts/full_dataset_audit.py --execute --concurrency 2
    python scripts/full_dataset_audit.py --execute --max-persons 3 \
        --max-companies 3 --max-locations 3 --max-pairs 5 \
        --max-nary-triples 2 --max-nary-fives 1 --max-nary-tens 0
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
from typing import Any, Callable, Iterable, Mapping, Sequence
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
NARY_UNION_TEMPLATES = (
    "{entities}分别有哪些关联公司？请合并结果。",
    "请查询{entities}各自关联的企业，并给出并集。",
    "综合来看，{entities}连接到哪些公司？",
)
PROFILE_TEMPLATES = (
    "请展示{entity}的基础资料。",
    "我想了解{entity}的实体信息。",
)
NARY_INTERSECTION_TEMPLATES = (
    "{entities}共同关联到哪些公司？",
    "请找出同时与{entities}存在直接关联的企业。",
)
NARY_DIRECT_TEMPLATES = (
    "{entities}之间有哪些直接关系？只看这些实体内部。",
    "请给出{entities}形成的内部直接关系图，不扩展一跳邻居。",
    "仅查询{entities}彼此之间的关系。",
)
MULTI_GOAL_TEMPLATE = "请分别查询{entity}明确持有的公司，以及其任职的公司。"
DEPENDENT_LOCATION_TEMPLATE = (
    "先找出{entity}直接关联的全部公司，再查询这些公司的总部地点。"
)
PROFILE_SUITES = frozenset({"person_profiles", "company_profiles"})
SEMANTIC_SUITES = frozenset(
    {
        "nary_intersection_3",
        "nary_intersection_5",
        "nary_direct_3",
        "nary_direct_5",
        "nary_direct_10",
        "multi_goal_empty_nonempty",
        "dependent_goal_location",
    }
)
TRACE_COUNTER_FIELDS = (
    "model_calls",
    "planner_model_calls",
    "researcher_model_calls",
    "visualizer_model_calls",
    "tool_calls",
    "research_steps",
    "replans",
)
REPORT_RESULT_FIELDS = frozenset(
    {
        "case_id",
        "suite",
        "passed",
        "error",
        "request_id",
        "response_status",
        "response_error_code",
        "expected_nodes",
        "expected_edges",
        "actual_nodes",
        "actual_edges",
        "cache_hit",
        "cache_match_type",
        "focus_company_count",
        "tool_result_counts",
        "verified_relation_rows",
        *TRACE_COUNTER_FIELDS,
    }
)
REPORT_AGENT_STEP_FIELDS = frozenset(
    {
        "role",
        "action",
        "tool",
        "association_operator",
        "relation_types",
        "result_merge",
        "resolution_strategy",
        "resolution_version",
        "record_ids",
        "argument_fingerprint",
        "count",
        "error_code",
        "goal_id",
    }
)
LOCATION_FOLLOWUP_TEMPLATES = (
    "这些公司在哪？",
    "它们分别位于哪些城市？",
    "上述关联企业的总部在哪里？",
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
    expected_cache_hit: bool | None = None
    expected_result_merge: str | None = None
    seed_entity_ids: tuple[str, ...] = ()
    focus_company_ids: tuple[str, ...] = ()
    require_batch_entity_tools: bool = False
    expected_tool_calls: int | None = None
    expected_tool_result_counts: tuple[tuple[str, int], ...] = ()
    expected_relation_scopes: tuple[tuple[str, ...], ...] = ()
    relation_result_required: bool = True

    @property
    def is_empty(self) -> bool:
        return self.relation_result_required and not self.expected_edges


@dataclass(frozen=True)
class AuditConversation:
    conversation_id: str
    entity_count: int
    seed_entity_ids: tuple[str, ...]
    steps: tuple[AuditCase, ...]


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

    def build_extended_cases(
        self,
        *,
        max_person_profiles: int | None = None,
        max_company_profiles: int | None = None,
        max_intersections: int = 2,
        max_direct_groups: int = 3,
        max_multi_goal: int = 1,
        max_dependent_location: int = 1,
        skip_empty: bool = False,
    ) -> tuple[AuditCase, ...]:
        """Build profile and non-union semantic cases from the raw graph.

        The limits bound paid HTTP work only.  Entity membership, result rows, and
        expected provenance are always derived from the three raw arrays; no
        acceptance entity or source row is named here.
        """

        limits = (
            max_intersections,
            max_direct_groups,
            max_multi_goal,
            max_dependent_location,
        )
        if any(limit < 0 for limit in limits):
            raise AuditFailure("extended_limit_must_be_nonnegative")
        profiles = (
            *self._person_profile_cases()[:max_person_profiles],
            *self._company_profile_cases()[:max_company_profiles],
        )
        semantic = (
            *self._intersection_cases()[:max_intersections],
            *self._direct_cases()[:max_direct_groups],
            *self._multi_goal_cases()[:max_multi_goal],
            *self._dependent_location_cases()[:max_dependent_location],
        )
        cases = (*profiles, *semantic)
        if skip_empty:
            cases = tuple(case for case in cases if not case.is_empty)
        return tuple(cases)

    def build_nary_conversations(
        self,
        *,
        max_triples: int = 20,
        max_fives: int = 10,
        max_tens: int = 2,
    ) -> tuple[AuditConversation, ...]:
        """Build deterministic N-entity union/follow-up/cache conversations.

        Entity membership comes only from a stable walk over the raw business
        relationship graph.  Query wording changes presentation, never expected
        facts.  Every conversation remains sequential while separate conversations
        may be executed concurrently.
        """

        limits = ((3, max_triples), (5, max_fives), (10, max_tens))
        if any(limit < 0 for _, limit in limits):
            raise AuditFailure("nary_limit_must_be_nonnegative")
        walk = self._business_graph_walk()
        output: list[AuditConversation] = []
        for entity_count, limit in limits:
            for number, seed_ids in enumerate(
                self._graph_groups(walk, entity_count, limit),
                start=1,
            ):
                union_edges, focus_company_ids = self._union_projection(seed_ids)
                if not union_edges or not focus_company_ids:
                    raise AuditFailure(
                        f"nary_union_must_be_nonempty:{entity_count}:{number}"
                    )
                location_edges = self._headquarters_edges(focus_company_ids)
                union_nodes = self._case_nodes(seed_ids, union_edges)
                session_edges = tuple(sorted({*union_edges, *location_edges}))
                session_nodes = self._case_nodes(seed_ids, session_edges)
                labels = tuple(self.nodes[entity_id].label for entity_id in seed_ids)
                union_message = _nary_union_message(labels, number - 1)
                location_message = _location_followup_message(number - 1)
                required_entity_tools = {
                    "persons" if entity_id.startswith("person:") else "companies"
                    for entity_id in seed_ids
                }
                required_tools = tuple(sorted({*required_entity_tools, "relations"}))
                prefix = f"nary-{entity_count}-{number:03d}"
                union_case = AuditCase(
                    case_id=f"{prefix}-union",
                    suite=f"nary_union_{entity_count}",
                    message=union_message,
                    expected_nodes=union_nodes,
                    expected_edges=union_edges,
                    required_tools=required_tools,
                    expected_cache_hit=False,
                    expected_result_merge="union",
                    seed_entity_ids=seed_ids,
                    focus_company_ids=focus_company_ids,
                    require_batch_entity_tools=True,
                    expected_tool_calls=len(required_tools),
                )
                location_case = AuditCase(
                    case_id=f"{prefix}-locations",
                    suite=f"nary_locations_{entity_count}",
                    message=location_message,
                    expected_nodes=session_nodes,
                    expected_edges=session_edges,
                    required_tools=("relations",),
                    expected_cache_hit=False,
                    expected_result_merge="not_applicable",
                    seed_entity_ids=focus_company_ids,
                    focus_company_ids=focus_company_ids,
                    expected_tool_calls=1,
                )
                repeat_case = AuditCase(
                    case_id=f"{prefix}-raw-repeat",
                    suite=f"nary_cache_{entity_count}",
                    message=union_message,
                    expected_nodes=session_nodes,
                    expected_edges=session_edges,
                    required_tools=(),
                    expected_cache_hit=True,
                    expected_result_merge=None,
                    seed_entity_ids=seed_ids,
                    focus_company_ids=focus_company_ids,
                    expected_tool_calls=0,
                )
                output.append(
                    AuditConversation(
                        conversation_id=prefix,
                        entity_count=entity_count,
                        seed_entity_ids=seed_ids,
                        steps=(union_case, location_case, repeat_case),
                    )
                )
        return tuple(output)

    def _business_graph_walk(self) -> tuple[str, ...]:
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            if edge.raw_relation not in BUSINESS_RELATIONS:
                continue
            if (
                edge.source not in self.base_entity_ids
                or edge.target not in self.base_entity_ids
            ):
                continue
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        if len(adjacency) < 10:
            raise AuditFailure("business_graph_has_too_few_entities")

        order: list[str] = []
        unseen = set(adjacency)
        while unseen:
            start = min(unseen, key=lambda item: (-len(adjacency[item]), item))
            queue = [start]
            queued = {start}
            while queue:
                current = queue.pop(0)
                if current not in unseen:
                    continue
                unseen.remove(current)
                order.append(current)
                neighbours = sorted(
                    (adjacency[current] & unseen) - queued,
                    key=lambda item: (-len(adjacency[item]), item),
                )
                queue.extend(neighbours)
                queued.update(neighbours)
        return tuple(order)

    @staticmethod
    def _graph_groups(
        walk: Sequence[str],
        entity_count: int,
        limit: int,
    ) -> tuple[tuple[str, ...], ...]:
        if limit == 0:
            return ()
        if entity_count < 1 or len(walk) < entity_count:
            raise AuditFailure(f"nary_group_size_unavailable:{entity_count}")
        groups: list[tuple[str, ...]] = []
        seen: set[frozenset[str]] = set()
        for start in range(len(walk)):
            group = tuple(
                walk[(start + offset) % len(walk)]
                for offset in range(entity_count)
            )
            key = frozenset(group)
            if len(key) != entity_count or key in seen:
                continue
            seen.add(key)
            groups.append(group)
            if len(groups) == limit:
                return tuple(groups)
        raise AuditFailure(
            f"nary_group_count_unavailable:{entity_count}:{limit}:{len(groups)}"
        )

    def _union_projection(
        self,
        seed_ids: Sequence[str],
    ) -> tuple[tuple[ExpectedEdge, ...], tuple[str, ...]]:
        selected_edges: set[ExpectedEdge] = set()
        focus_company_ids: set[str] = set()
        for subject_id in seed_ids:
            for edge in self.edges:
                if (
                    edge.raw_relation not in BUSINESS_RELATIONS
                    or subject_id not in {edge.source, edge.target}
                ):
                    continue
                opposite = edge.target if edge.source == subject_id else edge.source
                if not opposite.startswith("company:"):
                    continue
                selected_edges.add(edge)
                focus_company_ids.add(opposite)
        return tuple(sorted(selected_edges)), tuple(sorted(focus_company_ids))

    def _headquarters_edges(
        self,
        company_ids: Iterable[str],
    ) -> tuple[ExpectedEdge, ...]:
        selected = set(company_ids)
        return tuple(
            edge
            for edge in self.edges
            if edge.raw_relation == "Headquartered_in" and edge.source in selected
        )

    def _case_nodes(
        self,
        seed_ids: Iterable[str],
        edges: Iterable[ExpectedEdge],
    ) -> tuple[ExpectedNode, ...]:
        ids = set(seed_ids)
        for edge in edges:
            ids.update((edge.source, edge.target))
        return tuple(sorted(self.nodes[entity_id] for entity_id in ids))

    @staticmethod
    def _entity_tool(entity_id: str) -> str:
        return "persons" if entity_id.startswith("person:") else "companies"

    def _required_tools(self, entity_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted({*(self._entity_tool(entity_id) for entity_id in entity_ids), "relations"})
        )

    def _company_neighbour_edges(
        self, subject_id: str
    ) -> dict[str, tuple[ExpectedEdge, ...]]:
        grouped: defaultdict[str, list[ExpectedEdge]] = defaultdict(list)
        for edge in self.edges:
            if (
                edge.raw_relation not in BUSINESS_RELATIONS
                or subject_id not in {edge.source, edge.target}
            ):
                continue
            neighbour = edge.target if edge.source == subject_id else edge.source
            if neighbour.startswith("company:"):
                grouped[neighbour].append(edge)
        return {
            entity_id: tuple(sorted(rows))
            for entity_id, rows in grouped.items()
        }

    def _person_profile_cases(self) -> tuple[AuditCase, ...]:
        return tuple(
            self._profile_case(
                entity_id=f"person:{row['id']}",
                suite="person_profiles",
                variant=index,
            )
            for index, row in enumerate(self.persons)
        )

    def _company_profile_cases(self) -> tuple[AuditCase, ...]:
        return tuple(
            self._profile_case(
                entity_id=f"company:{row['id']}",
                suite="company_profiles",
                variant=index,
            )
            for index, row in enumerate(self.companies)
        )

    def _profile_case(
        self,
        *,
        entity_id: str,
        suite: str,
        variant: int,
    ) -> AuditCase:
        tool = self._entity_tool(entity_id)
        node = self.nodes[entity_id]
        return AuditCase(
            case_id=f"profile-{entity_id.replace(':', '-')}",
            suite=suite,
            message=PROFILE_TEMPLATES[variant % len(PROFILE_TEMPLATES)].format(
                entity=node.label
            ),
            expected_nodes=(node,),
            expected_edges=(),
            required_tools=(tool,),
            expected_cache_hit=False,
            expected_result_merge="not_applicable",
            seed_entity_ids=(entity_id,),
            require_batch_entity_tools=True,
            expected_tool_calls=1,
            expected_tool_result_counts=((tool, 1),),
            relation_result_required=False,
        )

    def _intersection_cases(self) -> tuple[AuditCase, ...]:
        cases: list[AuditCase] = []
        # The raw graph has common-neighbour sets large enough for N=3 and N=5.
        # The selection algorithm remains valid for any future source data and
        # fails explicitly if the raw graph no longer supports the requested size.
        for variant, entity_count in enumerate((3, 5)):
            candidates: list[tuple[int, str, tuple[str, ...]]] = []
            for company_id in sorted(
                entity_id
                for entity_id in self.base_entity_ids
                if entity_id.startswith("company:")
            ):
                subjects = tuple(
                    sorted(
                        subject_id
                        for subject_id in self.base_entity_ids
                        if subject_id != company_id
                        and company_id in self._company_neighbour_edges(subject_id)
                    )
                )
                if len(subjects) >= entity_count:
                    candidates.append((-len(subjects), company_id, subjects))
            if not candidates:
                raise AuditFailure(f"intersection_group_unavailable:{entity_count}")
            _rank, _target, available = min(candidates)
            seed_ids = available[:entity_count]
            memberships = [self._company_neighbour_edges(item) for item in seed_ids]
            common = set.intersection(*(set(item) for item in memberships))
            selected_edges = tuple(
                sorted(
                    {
                        edge
                        for item in memberships
                        for company_id, rows in item.items()
                        if company_id in common
                        for edge in rows
                    }
                )
            )
            if not common or not selected_edges:
                raise AuditFailure(f"intersection_projection_empty:{entity_count}")
            labels = [self.nodes[entity_id].label for entity_id in seed_ids]
            required_tools = self._required_tools(seed_ids)
            entity_tool_counts = tuple(
                (tool, 1) for tool in required_tools if tool != "relations"
            )
            cases.append(
                AuditCase(
                    case_id=f"intersection-{entity_count}",
                    suite=f"nary_intersection_{entity_count}",
                    message=NARY_INTERSECTION_TEMPLATES[
                        variant % len(NARY_INTERSECTION_TEMPLATES)
                    ].format(entities="、".join(labels)),
                    expected_nodes=self._case_nodes(seed_ids, selected_edges),
                    expected_edges=selected_edges,
                    required_tools=required_tools,
                    expected_cache_hit=False,
                    expected_result_merge="intersection",
                    seed_entity_ids=seed_ids,
                    focus_company_ids=tuple(sorted(common)),
                    require_batch_entity_tools=True,
                    expected_tool_calls=len(required_tools),
                    expected_tool_result_counts=(*entity_tool_counts, ("relations", 1)),
                    expected_relation_scopes=((),),
                )
            )
        return tuple(cases)

    def _direct_cases(self) -> tuple[AuditCase, ...]:
        walk = self._business_graph_walk()
        business_edges = [
            edge
            for edge in self.edges
            if edge.raw_relation in BUSINESS_RELATIONS
            and edge.source in self.base_entity_ids
            and edge.target in self.base_entity_ids
            and edge.source != edge.target
        ]
        if not business_edges:
            raise AuditFailure("direct_seed_edge_unavailable")
        cases: list[AuditCase] = []
        used: set[frozenset[str]] = set()
        for variant, entity_count in enumerate((3, 5, 10)):
            chosen: tuple[str, ...] | None = None
            for seed_edge in business_edges[variant:] + business_edges[:variant]:
                ordered = [seed_edge.source, seed_edge.target]
                ordered.extend(item for item in walk if item not in ordered)
                candidate = tuple(ordered[:entity_count])
                key = frozenset(candidate)
                if len(key) == entity_count and key not in used:
                    chosen = candidate
                    used.add(key)
                    break
            if chosen is None:
                raise AuditFailure(f"direct_group_unavailable:{entity_count}")
            selected_edges = tuple(
                edge
                for edge in self.edges
                if edge.raw_relation in BUSINESS_RELATIONS
                and edge.source in chosen
                and edge.target in chosen
            )
            if not selected_edges:
                raise AuditFailure(f"direct_projection_empty:{entity_count}")
            required_tools = self._required_tools(chosen)
            entity_tool_counts = tuple(
                (tool, 1) for tool in required_tools if tool != "relations"
            )
            cases.append(
                AuditCase(
                    case_id=f"direct-{entity_count}",
                    suite=f"nary_direct_{entity_count}",
                    message=NARY_DIRECT_TEMPLATES[
                        variant % len(NARY_DIRECT_TEMPLATES)
                    ].format(
                        entities="、".join(self.nodes[item].label for item in chosen)
                    ),
                    expected_nodes=self._case_nodes(chosen, selected_edges),
                    expected_edges=selected_edges,
                    required_tools=required_tools,
                    expected_cache_hit=False,
                    expected_result_merge="direct",
                    seed_entity_ids=chosen,
                    focus_company_ids=tuple(
                        sorted(item for item in chosen if item.startswith("company:"))
                    ),
                    require_batch_entity_tools=True,
                    expected_tool_calls=len(required_tools),
                    expected_tool_result_counts=(*entity_tool_counts, ("relations", 1)),
                    expected_relation_scopes=((),),
                )
            )
        return tuple(cases)

    def _multi_goal_cases(self) -> tuple[AuditCase, ...]:
        cases: list[AuditCase] = []
        for row in self.persons:
            subject_id = f"person:{row['id']}"
            role_edges = tuple(
                edge
                for edge in self.edges
                if edge.source == subject_id
                and edge.relation_type == "works_at"
                and edge.target.startswith("company:")
            )
            owns_edges = tuple(
                edge
                for edge in self.edges
                if edge.source == subject_id and edge.relation_type == "owns"
            )
            if not role_edges or owns_edges:
                continue
            cases.append(
                AuditCase(
                    case_id=f"multi-goal-empty-nonempty-{row['id']}",
                    suite="multi_goal_empty_nonempty",
                    message=MULTI_GOAL_TEMPLATE.format(entity=row["name"]),
                    expected_nodes=self._case_nodes((subject_id,), role_edges),
                    expected_edges=role_edges,
                    required_tools=("persons", "relations"),
                    expected_cache_hit=False,
                    expected_result_merge="not_applicable",
                    seed_entity_ids=(subject_id,),
                    require_batch_entity_tools=True,
                    expected_tool_calls=3,
                    expected_tool_result_counts=(("persons", 1), ("relations", 2)),
                    expected_relation_scopes=(("owns",), ("works_at",)),
                )
            )
        if not cases:
            raise AuditFailure("multi_goal_empty_nonempty_unavailable")
        return tuple(cases)

    def _dependent_location_cases(self) -> tuple[AuditCase, ...]:
        cases: list[AuditCase] = []
        for row in self.persons:
            subject_id = f"person:{row['id']}"
            business_edges, company_ids = self._union_projection((subject_id,))
            location_edges = self._headquarters_edges(company_ids)
            if not business_edges or not company_ids or not location_edges:
                continue
            selected_edges = tuple(sorted({*business_edges, *location_edges}))
            cases.append(
                AuditCase(
                    case_id=f"dependent-location-{row['id']}",
                    suite="dependent_goal_location",
                    message=DEPENDENT_LOCATION_TEMPLATE.format(entity=row["name"]),
                    expected_nodes=self._case_nodes((subject_id,), selected_edges),
                    expected_edges=selected_edges,
                    required_tools=("persons", "relations"),
                    expected_cache_hit=False,
                    expected_result_merge="not_applicable",
                    seed_entity_ids=(subject_id,),
                    focus_company_ids=company_ids,
                    require_batch_entity_tools=True,
                    expected_tool_calls=3,
                    expected_tool_result_counts=(("persons", 1), ("relations", 2)),
                    expected_relation_scopes=((), ("headquartered_in",)),
                )
            )
        if not cases:
            raise AuditFailure("dependent_location_goal_unavailable")
        return tuple(cases)

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
        for number, (pair, _seed_rows) in enumerate(
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
            # ``direct`` is an induced-subgraph operation: once the seed set is
            # chosen, every business edge whose two endpoints belong to that
            # set is part of the result.  Re-filter from the lossless raw edge
            # projection instead of reusing the endpoint-group rows that only
            # established which pair cases exist.  This deliberately retains
            # duplicate rows and self-relations on either seed.
            seed_ids = frozenset(ids)
            edges = tuple(
                sorted(
                    edge
                    for edge in self.edges
                    if edge.raw_relation in BUSINESS_RELATIONS
                    and edge.source in seed_ids
                    and edge.target in seed_ids
                )
            )
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

    def chat(
        self,
        case: AuditCase,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/chat",
            {
                "conversation_id": conversation_id or str(uuid4()),
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
        for key in TRACE_COUNTER_FIELDS
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
    route_history = trace.get("route_history")
    if not isinstance(route_history, list):
        raise AuditFailure("missing_route_history")

    memory = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    cache_hit = memory.get("cache_hit") is True
    if case.expected_cache_hit is not None and cache_hit is not case.expected_cache_hit:
        raise AuditFailure(
            f"cache_expectation_mismatch:expected={case.expected_cache_hit}:actual={cache_hit}"
        )
    tool_result_counts: Counter[str] = Counter()
    if cache_hit:
        if counters["model_calls"] != 0 or counters["tool_calls"] != 0:
            raise AuditFailure("cache_hit_executed_agent_work")
        if memory.get("match_type") != "raw_exact":
            raise AuditFailure("repeat_query_did_not_use_raw_exact_cache")
    else:
        if (
            trace.get("researcher_invoked") is not True
            or not 2 <= counters["planner_model_calls"] <= 4
            or counters["researcher_model_calls"] < 1
            or counters["visualizer_model_calls"] < 1
            or counters["tool_calls"] < 1
        ):
            raise AuditFailure("fresh_trace_missing_agent_work")
        analysis_indexes = [
            index
            for index, route in enumerate(route_history)
            if route == "planner_analyze"
        ]
        task_indexes = [
            index
            for index, route in enumerate(route_history)
            if route == "planner_tasks"
        ]
        if not analysis_indexes:
            raise AuditFailure("fresh_trace_missing_planner_analysis")
        if not task_indexes or min(task_indexes) <= min(analysis_indexes):
            raise AuditFailure("fresh_trace_missing_planner_tasks")
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
        tool_result_steps = [
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("role") == "researcher"
            and step.get("action") == "tool_result"
            and step.get("tool")
        ]
        tool_result_counts.update(str(step["tool"]) for step in tool_result_steps)
        if case.expected_tool_result_counts:
            actual_counts = Counter(str(step["tool"]) for step in tool_result_steps)
            expected_counts = Counter(dict(case.expected_tool_result_counts))
            if actual_counts != expected_counts:
                raise AuditFailure(
                    "tool_result_counts_mismatch:"
                    f"expected={_counter_text(expected_counts)}:"
                    f"actual={_counter_text(actual_counts)}"
                )
        if case.expected_relation_scopes:
            actual_scopes = Counter(
                tuple(sorted(_optional_string_list(step.get("relation_types"))))
                for step in tool_result_steps
                if step.get("tool") == "relations"
            )
            expected_scopes = Counter(case.expected_relation_scopes)
            if actual_scopes != expected_scopes:
                raise AuditFailure("relation_tool_scopes_mismatch")
        if (
            case.expected_tool_calls is not None
            and counters["tool_calls"] != case.expected_tool_calls
        ):
            raise AuditFailure(
                f"tool_call_count_mismatch:{case.expected_tool_calls}:{counters['tool_calls']}"
            )
        if case.expected_result_merge is not None:
            planner_steps = [
                step
                for step in steps
                if isinstance(step, dict)
                and step.get("role") == "planner"
                and step.get("action") == "plan"
            ]
            if len(planner_steps) != 1 or planner_steps[0].get(
                "result_merge"
            ) != case.expected_result_merge:
                raise AuditFailure("planner_result_merge_mismatch")
        if case.require_batch_entity_tools:
            _validate_batch_entity_trace(case, steps)

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
        "focus_company_count": len(case.focus_company_ids),
        "tool_result_counts": dict(sorted(tool_result_counts.items())),
        "verified_relation_rows": sorted(
            {edge.source_row for edge in case.expected_edges}
        ),
        **counters,
    }


def execute_plan(
    *,
    client: ApiClient,
    cases: Sequence[AuditCase],
    concurrency: int,
    on_case_complete: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run independent-conversation cases concurrently and preserve plan order."""

    results: list[dict[str, Any] | None] = [None] * len(cases)

    def run_one(index: int, case: AuditCase) -> tuple[int, dict[str, Any]]:
        body: dict[str, Any] | None = None
        try:
            body = client.chat(case)
            return index, validate_response(case, body)
        except Exception as exc:  # every case should leave a bounded report row
            return index, _failure_row(case, exc, body)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: list[Future[tuple[int, dict[str, Any]]]] = [
            executor.submit(run_one, index, case)
            for index, case in enumerate(cases)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            if on_case_complete is not None:
                on_case_complete(index, result)
            outcome = "PASS" if result["passed"] else "FAIL"
            print(f"{outcome} {result['case_id']}", file=sys.stderr, flush=True)
    return [result for result in results if result is not None]


def execute_conversations(
    *,
    client: ApiClient,
    conversations: Sequence[AuditConversation],
    concurrency: int,
    on_conversation_complete: (
        Callable[[int, list[dict[str, Any]]], None] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Run each conversation in order while parallelizing only across conversations."""

    results: list[list[dict[str, Any]] | None] = [None] * len(conversations)

    def run_one(
        index: int,
        conversation: AuditConversation,
    ) -> tuple[int, list[dict[str, Any]]]:
        conversation_id = str(uuid4())
        rows: list[dict[str, Any]] = []
        for case in conversation.steps:
            body: dict[str, Any] | None = None
            try:
                body = client.chat(case, conversation_id=conversation_id)
                row = validate_response(case, body)
            except Exception as exc:
                row = _failure_row(case, exc, body)
            rows.append(row)
            outcome = "PASS" if row["passed"] else "FAIL"
            print(f"{outcome} {row['case_id']}", file=sys.stderr, flush=True)
        return index, rows

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: list[Future[tuple[int, list[dict[str, Any]]]]] = [
            executor.submit(run_one, index, conversation)
            for index, conversation in enumerate(conversations)
        ]
        for future in as_completed(futures):
            index, rows = future.result()
            results[index] = rows
            if on_conversation_complete is not None:
                # Publish only an entirely attempted conversation.  A process
                # interruption midway through the three-turn sequence must not
                # make a later run look as though its follow-up/cache semantics
                # were verified.
                on_conversation_complete(index, rows)
    return [row for rows in results if rows is not None for row in rows]


def _failure_row(
    case: AuditCase,
    error: Exception,
    body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep enough safe execution metadata to diagnose a failed live case."""

    response = body if isinstance(body, Mapping) else {}
    trace = response.get("trace")
    safe_trace: dict[str, Any] = {}
    if isinstance(trace, Mapping):
        for key in (
            "model_calls",
            "planner_model_calls",
            "researcher_model_calls",
            "visualizer_model_calls",
            "tool_calls",
            "research_steps",
            "replans",
            "route_history",
            "agent_steps",
        ):
            if key in trace:
                safe_trace[key] = trace[key]
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "passed": False,
        "error": _redact(str(error))[:500] or type(error).__name__,
        "request_id": response.get("request_id"),
        "response_status": response.get("status"),
        "response_error_code": response.get("error_code"),
        "trace": safe_trace,
        "expected_nodes": len(case.expected_nodes),
        "expected_edges": len(case.expected_edges),
        "tool_result_counts": _tool_result_counts(safe_trace.get("agent_steps")),
        "verified_relation_rows": [],
    }


def _validate_batch_entity_trace(
    case: AuditCase,
    steps: Sequence[Any],
) -> None:
    planner_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("role") == "planner"
        and step.get("action") == "plan"
    ]
    if len(planner_steps) != 1 or planner_steps[0].get("count") != len(
        case.seed_entity_ids
    ):
        raise AuditFailure("planner_entity_count_mismatch")

    expected_by_tool = {
        "persons": {
            entity_id
            for entity_id in case.seed_entity_ids
            if entity_id.startswith("person:")
        },
        "companies": {
            entity_id
            for entity_id in case.seed_entity_ids
            if entity_id.startswith("company:")
        },
    }
    for tool, expected_ids in expected_by_tool.items():
        if not expected_ids:
            continue
        results = [
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("role") == "researcher"
            and step.get("action") == "tool_result"
            and step.get("tool") == tool
        ]
        if len(results) != 1:
            raise AuditFailure(f"entity_tool_not_batched:{tool}:{len(results)}")
        result = results[0]
        if result.get("resolution_strategy") != "exact":
            raise AuditFailure(f"batch_entity_resolution_not_exact:{tool}")
        if expected_ids - set(_optional_string_list(result.get("record_ids"))):
            raise AuditFailure(f"batch_entity_ids_incomplete:{tool}")


def _tool_result_counts(steps: Any) -> dict[str, int]:
    if not isinstance(steps, list):
        return {}
    counts = Counter(
        str(step["tool"])
        for step in steps
        if isinstance(step, Mapping)
        and step.get("role") == "researcher"
        and step.get("action") == "tool_result"
        and step.get("tool") in {"persons", "companies", "relations"}
    )
    return dict(sorted(counts.items()))


def _result_counter(result: Mapping[str, Any], field: str) -> int:
    value = result.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    trace = result.get("trace")
    if isinstance(trace, Mapping):
        value = trace.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _report_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist one result row before it reaches a persisted report."""

    safe = {key: result[key] for key in REPORT_RESULT_FIELDS if key in result}
    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        return safe
    safe_trace = {
        key: trace[key]
        for key in (*TRACE_COUNTER_FIELDS, "route_history")
        if key in trace
    }
    steps = trace.get("agent_steps")
    if isinstance(steps, list):
        safe_trace["agent_steps"] = [
            {
                key: step[key]
                for key in REPORT_AGENT_STEP_FIELDS
                if key in step
            }
            for step in steps
            if isinstance(step, Mapping)
        ]
    safe["trace"] = safe_trace
    return safe


def build_execution_report(
    *,
    base_url: str,
    dataset: RawDataset,
    cases: Sequence[AuditCase],
    conversations: Sequence[AuditConversation],
    independent_results: Sequence[Mapping[str, Any] | None],
    conversation_results: Sequence[Sequence[Mapping[str, Any]] | None],
    phase: str,
) -> dict[str, Any]:
    """Build a query-free, interruption-safe audit progress snapshot.

    Only whole N-ary conversations are supplied in ``conversation_results``.
    This keeps a partial report from claiming that multi-turn context/cache
    semantics were verified when a process stopped between conversation steps.
    """

    if len(independent_results) != len(cases):
        raise AuditFailure("independent_progress_shape_mismatch")
    if len(conversation_results) != len(conversations):
        raise AuditFailure("conversation_progress_shape_mismatch")

    completed: list[Mapping[str, Any]] = []
    for case, result in zip(cases, independent_results, strict=True):
        if result is None:
            continue
        if (
            result.get("case_id") != case.case_id
            or result.get("suite") != case.suite
        ):
            raise AuditFailure("independent_progress_case_mismatch")
        completed.append(_report_result(result))

    completed_conversations = 0
    for conversation, rows in zip(conversations, conversation_results, strict=True):
        if rows is None:
            continue
        expected_ids = [step.case_id for step in conversation.steps]
        actual_ids = [row.get("case_id") for row in rows]
        expected_suites = [step.suite for step in conversation.steps]
        actual_suites = [row.get("suite") for row in rows]
        if actual_ids != expected_ids or actual_suites != expected_suites:
            raise AuditFailure("conversation_progress_case_mismatch")
        completed_conversations += 1
        completed.extend(_report_result(row) for row in rows)

    planned_cases = [
        *cases,
        *(step for conversation in conversations for step in conversation.steps),
    ]
    planned_count = len(planned_cases)
    completed_count = len(completed)
    is_complete = (
        completed_count == planned_count
        and completed_conversations == len(conversations)
    )
    passed = sum(result.get("passed") is True for result in completed)
    failed = completed_count - passed

    planned_suites = Counter(case.suite for case in planned_cases)
    completed_suites = Counter(str(result.get("suite")) for result in completed)
    passed_suites = Counter(
        str(result.get("suite"))
        for result in completed
        if result.get("passed") is True
    )
    failed_suites = completed_suites - passed_suites
    suite_results = {
        suite: {
            "planned": planned_suites[suite],
            "completed": completed_suites[suite],
            "passed": passed_suites[suite],
            "failed": failed_suites[suite],
            "remaining": planned_suites[suite] - completed_suites[suite],
        }
        for suite in sorted(planned_suites)
    }

    execution_counts = {
        field: sum(_result_counter(result, field) for result in completed)
        for field in TRACE_COUNTER_FIELDS
    }
    fact_tool_results: Counter[str] = Counter()
    for result in completed:
        values = result.get("tool_result_counts")
        if not isinstance(values, Mapping):
            continue
        for tool in ("persons", "companies", "relations"):
            count = values.get(tool)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                fact_tool_results[tool] += count

    cache_hit_count = sum(result.get("cache_hit") is True for result in completed)
    cache_miss_count = sum(result.get("cache_hit") is False for result in completed)
    cache_match_types = Counter(
        str(result["cache_match_type"])
        for result in completed
        if isinstance(result.get("cache_match_type"), str)
        and result.get("cache_match_type")
    )
    covered_relation_rows = sorted(
        {
            row
            for result in completed
            if result.get("passed") is True
            for row in result.get("verified_relation_rows", [])
            if isinstance(row, int) and not isinstance(row, bool) and row > 0
        }
    )

    case_suite_counts = Counter(case.suite for case in cases)
    nary_suite_counts = Counter(
        step.suite
        for conversation in conversations
        for step in conversation.steps
    )
    return {
        "report_version": "full-dataset-audit-v2",
        "mode": "executed",
        "completion_status": "complete" if is_complete else "partial",
        "complete": is_complete,
        "audit_outcome": (
            "failed"
            if is_complete and failed
            else "passed"
            if is_complete
            else "in_progress"
        ),
        "phase": phase,
        "base_url": base_url,
        "raw_counts": {
            "persons": len(dataset.persons),
            "companies": len(dataset.companies),
            "relations": len(dataset.relations),
        },
        "case_count": len(cases),
        "profile_case_count": sum(case.suite in PROFILE_SUITES for case in cases),
        "semantic_case_count": sum(case.suite in SEMANTIC_SUITES for case in cases),
        "nary_conversation_count": len(conversations),
        "completed_nary_conversations": completed_conversations,
        "nary_step_count": sum(len(item.steps) for item in conversations),
        "planned_chat_requests": planned_count,
        "completed_chat_requests": completed_count,
        "remaining_chat_requests": planned_count - completed_count,
        "passed": passed,
        "failed": failed,
        "suite_counts": dict(sorted(case_suite_counts.items())),
        "nary_suite_counts": dict(sorted(nary_suite_counts.items())),
        "suite_results": suite_results,
        "agent_counts": {
            "model_calls": execution_counts["model_calls"],
            "planner_model_calls": execution_counts["planner_model_calls"],
            "researcher_model_calls": execution_counts["researcher_model_calls"],
            "visualizer_model_calls": execution_counts["visualizer_model_calls"],
            "research_steps": execution_counts["research_steps"],
            "replans": execution_counts["replans"],
        },
        "tool_counts": {
            "executed_calls": execution_counts["tool_calls"],
            "successful_receipts_by_tool": {
                tool: fact_tool_results[tool]
                for tool in ("persons", "companies", "relations")
            },
        },
        "cache_counts": {
            "hits": cache_hit_count,
            "misses": cache_miss_count,
            "unknown": completed_count - cache_hit_count - cache_miss_count,
            "match_types": dict(sorted(cache_match_types.items())),
        },
        "raw_relation_coverage": {
            "covered_source_rows": covered_relation_rows,
            "covered_count": len(covered_relation_rows),
            "total_source_rows": len(dataset.relations),
        },
        # Rows intentionally contain no query text, prompt, credential or model
        # provider payload.  Failed rows retain only the API's bounded safe trace.
        "results": list(completed),
    }


def write_report_atomic(path: Path, report: Mapping[str, Any]) -> None:
    """Durably replace a report using a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_safe_json(report) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def plan_summary(
    cases: Sequence[AuditCase],
    conversations: Sequence[AuditConversation],
    dataset: RawDataset,
) -> dict[str, Any]:
    counts = Counter(case.suite for case in cases)
    conversation_steps = [
        step for conversation in conversations for step in conversation.steps
    ]
    nary_counts = Counter(conversation.entity_count for conversation in conversations)
    profile_count = sum(case.suite in PROFILE_SUITES for case in cases)
    semantic_count = sum(case.suite in SEMANTIC_SUITES for case in cases)
    return {
        "mode": "plan-only",
        "network_requests": 0,
        "raw_counts": {
            "persons": len(dataset.persons),
            "companies": len(dataset.companies),
            "relations": len(dataset.relations),
        },
        "case_count": len(cases),
        "profile_case_count": profile_count,
        "semantic_case_count": semantic_count,
        "suite_counts": dict(sorted(counts.items())),
        "empty_expected_case_count": sum(case.is_empty for case in cases),
        "nary_conversation_count": len(conversations),
        "nary_conversation_counts": {
            str(size): nary_counts.get(size, 0) for size in (3, 5, 10)
        },
        "nary_step_count": len(conversation_steps),
        "planned_chat_requests": len(cases) + len(conversation_steps),
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
        "--max-intersections",
        type=_nonnegative_int_argument,
        default=2,
    )
    parser.add_argument(
        "--max-direct-groups",
        type=_nonnegative_int_argument,
        default=3,
    )
    parser.add_argument(
        "--max-multi-goal",
        type=_nonnegative_int_argument,
        default=1,
    )
    parser.add_argument(
        "--max-dependent-location",
        type=_nonnegative_int_argument,
        default=1,
    )
    parser.add_argument(
        "--max-nary-triples",
        type=_nonnegative_int_argument,
        default=20,
    )
    parser.add_argument(
        "--max-nary-fives",
        type=_nonnegative_int_argument,
        default=10,
    )
    parser.add_argument(
        "--max-nary-tens",
        type=_nonnegative_int_argument,
        default=2,
    )
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
    base_cases = dataset.build_cases(
        max_persons=args.max_persons,
        max_companies=args.max_companies,
        max_locations=args.max_locations,
        max_pairs=args.max_pairs,
        skip_empty=args.skip_empty,
    )
    extended_cases = dataset.build_extended_cases(
        max_person_profiles=args.max_persons,
        max_company_profiles=args.max_companies,
        max_intersections=args.max_intersections,
        max_direct_groups=args.max_direct_groups,
        max_multi_goal=args.max_multi_goal,
        max_dependent_location=args.max_dependent_location,
        skip_empty=args.skip_empty,
    )
    cases = (*base_cases, *extended_cases)
    conversations = dataset.build_nary_conversations(
        max_triples=args.max_nary_triples,
        max_fives=args.max_nary_fives,
        max_tens=args.max_nary_tens,
    )
    if not args.execute:
        print(_safe_json(plan_summary(cases, conversations, dataset)))
        return 0
    if not cases and not conversations:
        raise SystemExit("execution plan is empty")

    client = ApiClient(args.base_url, args.timeout)
    for path in ("/health", "/ready"):
        status = client.get(path).get("status")
        if status not in {"ok", "ready", "healthy"}:
            raise AuditFailure(f"service_not_ready:{path}:{status}")

    independent_progress: list[dict[str, Any] | None] = [None] * len(cases)
    conversation_progress: list[list[dict[str, Any]] | None] = [
        None
    ] * len(conversations)

    def checkpoint(phase: str) -> dict[str, Any]:
        report = build_execution_report(
            base_url=client.base_url,
            dataset=dataset,
            cases=cases,
            conversations=conversations,
            independent_results=independent_progress,
            conversation_results=conversation_progress,
            phase=phase,
        )
        write_report_atomic(args.report, report)
        return report

    # Write a valid zero-progress snapshot before the first paid request.  Each
    # independent case then advances it from the coordinator thread; no worker
    # writes the file, so concurrency greater than one cannot interleave JSON.
    checkpoint("ready")

    def record_case(index: int, result: dict[str, Any]) -> None:
        independent_progress[index] = result
        checkpoint("independent_cases")

    execute_plan(
        client=client,
        cases=cases,
        concurrency=args.concurrency,
        on_case_complete=record_case,
    )
    checkpoint("independent_complete")

    def record_conversation(index: int, rows: list[dict[str, Any]]) -> None:
        conversation_progress[index] = rows
        checkpoint("nary_conversations")

    execute_conversations(
        client=client,
        conversations=conversations,
        concurrency=args.concurrency,
        on_conversation_complete=record_conversation,
    )
    report = checkpoint("complete")
    failures = [
        result
        for result in report["results"]
        if isinstance(result, Mapping) and result.get("passed") is not True
    ]
    print(
        _safe_json(
            {
                "mode": "executed",
                "case_count": len(cases),
                "nary_conversation_count": len(conversations),
                "planned_chat_requests": report["planned_chat_requests"],
                "completed_chat_requests": report["completed_chat_requests"],
                "passed": report["passed"],
                "failed": report["failed"],
                "report": str(args.report.resolve()),
            }
        )
    )
    return 1 if failures else 0


def _nary_union_message(labels: Sequence[str], variant: int) -> str:
    if not labels:
        raise AuditFailure("nary_query_requires_entities")
    return NARY_UNION_TEMPLATES[variant % len(NARY_UNION_TEMPLATES)].format(
        entities="、".join(labels)
    )


def _location_followup_message(variant: int) -> str:
    return LOCATION_FOLLOWUP_TEMPLATES[
        variant % len(LOCATION_FOLLOWUP_TEMPLATES)
    ]


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


def _optional_string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AuditFailure("invalid_optional_string_list")
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


def _counter_text(values: Counter[str]) -> str:
    return ",".join(
        f"{key}:{values[key]}" for key in sorted(values)
    ) or "none"


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
