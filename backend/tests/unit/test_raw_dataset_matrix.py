"""Data-driven coverage for every row in the three user-supplied JSON files.

This module deliberately computes its expectations from the immutable raw arrays.
It does not import a curated fixture, a bilingual alias table, or a production
projection constant.  The public ``RAW_ORACLE`` and its query-case helpers are also
usable by later scripted-StateGraph and opt-in live-model matrix tests.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from app.schemas import NodeType, ToolName
from app.tools import FixtureRepository, ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_FILES = ("person 1.json", "company 1.json", "relations 1.json")
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


def _load_array(path: Path) -> tuple[dict[str, Any], ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AssertionError(f"{path.name} must contain one JSON object array")
    return tuple(value)


def _slugify(value: str) -> str:
    """Independent form of the documented stable endpoint slug projection."""

    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(
        r"[^\w\u4e00-\u9fff]+", "-", normalized, flags=re.UNICODE
    )
    result = normalized.strip("-")
    if not result:
        raise AssertionError(f"raw endpoint cannot produce an empty slug: {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class EntityCase:
    entity_type: Literal["person", "company"]
    raw_id: str
    raw_name: str
    stable_id: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RelationCase:
    row_number: int
    raw_head: str
    raw_relation: str
    raw_tail: str
    source_id: str
    target_id: str

    @property
    def record_id(self) -> str:
        return f"relation:raw:{self.row_number:04d}"

    @property
    def evidence_id(self) -> str:
        return f"evidence:raw:relation:{self.row_number:04d}"


@dataclass(frozen=True, slots=True)
class QueryOracleCase:
    query: str
    subject_ids: tuple[str, ...]
    expected_company_ids: tuple[str, ...]
    expected_relation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectPairCase:
    query: str
    entity_ids: tuple[str, ...]
    expected_relation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawDatasetOracle:
    """Independent, read-only oracle derived solely from the three raw arrays."""

    directory: Path
    persons: tuple[dict[str, Any], ...]
    companies: tuple[dict[str, Any], ...]
    raw_relations: tuple[dict[str, Any], ...]
    entity_cases: tuple[EntityCase, ...]
    relation_cases: tuple[RelationCase, ...]
    stable_id_by_token: dict[str, str]
    label_by_stable_id: dict[str, str]
    entity_type_by_stable_id: dict[str, Literal["person", "company"]]

    @classmethod
    def load(cls, directory: Path) -> RawDatasetOracle:
        actual_json = {path.name for path in directory.glob("*.json")}
        if actual_json != set(RAW_FILES):
            raise AssertionError(
                f"data/ JSON set must be exactly {sorted(RAW_FILES)}; "
                f"got {sorted(actual_json)}"
            )

        persons = _load_array(directory / RAW_FILES[0])
        companies = _load_array(directory / RAW_FILES[1])
        raw_relations = _load_array(directory / RAW_FILES[2])
        if (len(persons), len(companies), len(raw_relations)) != (20, 30, 109):
            raise AssertionError(
                "raw source counts must remain 20 persons, 30 companies, and "
                f"109 relations; got {(len(persons), len(companies), len(raw_relations))}"
            )

        expected_keys = {
            "person": {"id", "name", "nationality", "summary"},
            "company": {"id", "name", "legal_rep_id", "city", "founded_year"},
            "relation": {"head", "relation", "tail"},
        }
        for kind, rows in (
            ("person", persons),
            ("company", companies),
            ("relation", raw_relations),
        ):
            for row_number, row in enumerate(rows, start=1):
                if set(row) != expected_keys[kind]:
                    raise AssertionError(
                        f"{kind} row {row_number} changed schema: {sorted(row)}"
                    )

        cases: list[EntityCase] = []
        stable_by_token: dict[str, str] = {}
        labels: dict[str, str] = {}
        types: dict[str, Literal["person", "company"]] = {}
        for entity_type, rows in (("person", persons), ("company", companies)):
            for row in rows:
                raw_id = str(row["id"])
                raw_name = str(row["name"])
                stable_id = f"{entity_type}:{raw_id}"
                case = EntityCase(entity_type, raw_id, raw_name, stable_id, row)
                cases.append(case)
                labels[stable_id] = raw_name
                types[stable_id] = entity_type
                for token in (raw_id, raw_name):
                    previous = stable_by_token.setdefault(token, stable_id)
                    if previous != stable_id:
                        raise AssertionError(f"ambiguous raw endpoint token: {token!r}")

        def endpoint_id(
            raw_value: str,
            raw_relation: str,
            position: Literal["head", "tail"],
        ) -> str:
            known = stable_by_token.get(raw_value)
            if known is not None:
                return known
            if raw_relation == "Headquartered_in" and position == "tail":
                return f"location:{_slugify(raw_value)}"
            is_person = raw_relation in ROLE_RELATIONS and position == "head"
            namespace = "person" if is_person else "company"
            return f"{namespace}:raw-reference:{_slugify(raw_value)}"

        relations = tuple(
            RelationCase(
                row_number=row_number,
                raw_head=str(row["head"]),
                raw_relation=str(row["relation"]),
                raw_tail=str(row["tail"]),
                source_id=endpoint_id(str(row["head"]), str(row["relation"]), "head"),
                target_id=endpoint_id(str(row["tail"]), str(row["relation"]), "tail"),
            )
            for row_number, row in enumerate(raw_relations, start=1)
        )
        return cls(
            directory=directory,
            persons=persons,
            companies=companies,
            raw_relations=raw_relations,
            entity_cases=tuple(cases),
            relation_cases=relations,
            stable_id_by_token=stable_by_token,
            label_by_stable_id=labels,
            entity_type_by_stable_id=types,
        )

    def relation_groups(
        self,
    ) -> dict[tuple[str, str, str], tuple[RelationCase, ...]]:
        groups: defaultdict[tuple[str, str, str], list[RelationCase]] = defaultdict(list)
        for case in self.relation_cases:
            groups[(case.source_id, case.raw_relation, case.target_id)].append(case)
        return {key: tuple(value) for key, value in groups.items()}

    def raw_reference_nodes(self) -> dict[str, tuple[str, str, int]]:
        """Return expected ID -> (entity type, label, first source row)."""

        expected: dict[str, tuple[str, str, int]] = {}
        for case in self.relation_cases:
            for stable_id, label in (
                (case.source_id, case.raw_head),
                (case.target_id, case.raw_tail),
            ):
                if ":raw-reference:" not in stable_id:
                    continue
                entity_type = stable_id.split(":", 1)[0]
                expected.setdefault(stable_id, (entity_type, label, case.row_number))
        return expected

    def person_company_queries(self) -> tuple[QueryOracleCase, ...]:
        """Build the broad single-person questions requested for StateGraph tests."""

        output: list[QueryOracleCase] = []
        for entity in self.entity_cases:
            if entity.entity_type != "person":
                continue
            matching = [
                relation
                for relation in self.relation_cases
                if relation.raw_relation != "Headquartered_in"
                and entity.stable_id in {relation.source_id, relation.target_id}
                and (
                    relation.target_id.startswith("company:")
                    or relation.source_id.startswith("company:")
                )
            ]
            company_ids = {
                endpoint
                for relation in matching
                for endpoint in (relation.source_id, relation.target_id)
                if endpoint.startswith("company:")
            }
            output.append(
                QueryOracleCase(
                    query=f"{entity.raw_name}有哪些公司？",
                    subject_ids=(entity.stable_id,),
                    expected_company_ids=tuple(sorted(company_ids)),
                    expected_relation_ids=tuple(
                        sorted(relation.record_id for relation in matching)
                    ),
                )
            )
        return tuple(output)

    def company_relation_queries(self) -> tuple[QueryOracleCase, ...]:
        """Build company-to-company one-hop questions without using model knowledge."""

        output: list[QueryOracleCase] = []
        for entity in self.entity_cases:
            if entity.entity_type != "company":
                continue
            matching = [
                relation
                for relation in self.relation_cases
                if relation.raw_relation != "Headquartered_in"
                and entity.stable_id in {relation.source_id, relation.target_id}
                and relation.source_id.startswith("company:")
                and relation.target_id.startswith("company:")
            ]
            company_ids = {
                endpoint
                for relation in matching
                for endpoint in (relation.source_id, relation.target_id)
                if endpoint != entity.stable_id
            }
            # A raw self-edge is still a verified company-to-company association.
            if any(
                relation.source_id == relation.target_id == entity.stable_id
                for relation in matching
            ):
                company_ids.add(entity.stable_id)
            output.append(
                QueryOracleCase(
                    query=f"{entity.raw_name}有哪些关联公司？",
                    subject_ids=(entity.stable_id,),
                    expected_company_ids=tuple(sorted(company_ids)),
                    expected_relation_ids=tuple(
                        sorted(relation.record_id for relation in matching)
                    ),
                )
            )
        return tuple(output)

    def direct_entity_pair_queries(self) -> tuple[DirectPairCase, ...]:
        """Group all business rows whose two endpoints exist in the entity arrays."""

        groups: defaultdict[frozenset[str], list[RelationCase]] = defaultdict(list)
        for relation in self.relation_cases:
            if relation.raw_relation == "Headquartered_in":
                continue
            if (
                relation.source_id not in self.label_by_stable_id
                or relation.target_id not in self.label_by_stable_id
            ):
                continue
            groups[frozenset((relation.source_id, relation.target_id))].append(relation)

        result: list[DirectPairCase] = []
        for pair, rows in groups.items():
            ids = tuple(sorted(pair))
            labels = [self.label_by_stable_id[item] for item in ids]
            if len(labels) == 1:
                query = f"{labels[0]}与其自身之间有什么关系？"
            else:
                query = f"{labels[0]}与{labels[1]}之间有什么关系？"
            result.append(
                DirectPairCase(
                    query=query,
                    entity_ids=ids,
                    expected_relation_ids=tuple(
                        sorted(relation.record_id for relation in rows)
                    ),
                )
            )
        return tuple(sorted(result, key=lambda item: item.query))


RAW_ORACLE = RawDatasetOracle.load(DATA_DIRECTORY)
ENTITY_PARAMS = [
    pytest.param(case, id=f"{case.entity_type}-{case.raw_id}")
    for case in RAW_ORACLE.entity_cases
]
COMPANY_PARAMS = [
    pytest.param(case, id=case.raw_id)
    for case in RAW_ORACLE.entity_cases
    if case.entity_type == "company"
]
RELATION_PARAMS = [
    pytest.param(
        case,
        id=f"row-{case.row_number:04d}-{case.raw_relation}",
    )
    for case in RAW_ORACLE.relation_cases
]


@pytest.fixture(scope="module")
def repository() -> FixtureRepository:
    return FixtureRepository.load(DATA_DIRECTORY)


@pytest.fixture(scope="module")
def registry(repository: FixtureRepository) -> ToolRegistry:
    return ToolRegistry(repository)


def _record_map(result: Any) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["record_kind"]), str(record["id"])): record
        for record in result.records
    }


def _assert_complete_evidence(result: Any) -> None:
    evidence_ids = {item.id for item in result.evidence}
    referenced_ids = {
        evidence_id
        for record in result.records
        for evidence_id in record["evidence_ids"]
    }
    assert evidence_ids == referenced_ids
    assert len(evidence_ids) == len(result.evidence)


def test_oracle_is_exclusively_derived_from_the_three_raw_arrays() -> None:
    assert len(RAW_ORACLE.entity_cases) == 50
    assert len(RAW_ORACLE.relation_cases) == 109
    assert len(RAW_ORACLE.person_company_queries()) == 20
    assert len(RAW_ORACLE.company_relation_queries()) == 30
    assert len(RAW_ORACLE.direct_entity_pair_queries()) == 57
    assert all(case.expected_company_ids for case in RAW_ORACLE.person_company_queries())


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ENTITY_PARAMS)
async def test_every_raw_entity_is_queryable_by_exact_name_and_raw_id(
    registry: ToolRegistry,
    case: EntityCase,
) -> None:
    if case.entity_type == "person":
        tool = ToolName.PERSONS
        id_field = "person_ids"
    else:
        tool = ToolName.COMPANIES
        id_field = "company_ids"

    by_name = await registry.execute(
        tool,
        {"query": case.raw_name, "match_mode": "exact"},
    )
    by_id = await registry.execute(tool, {id_field: [case.raw_id]})

    for result, expected_match_kind in (
        (by_name, "exact_name"),
        (by_id, "explicit_id"),
    ):
        assert result.success is True
        assert result.meta.total == result.meta.returned == 1
        assert result.meta.truncated is False
        records = _record_map(result)
        assert set(records) == {("entity", case.stable_id)}
        record = records[("entity", case.stable_id)]
        assert record["entity_type"] == case.entity_type
        assert record["label"] == case.raw_name
        assert record["properties"]["source_id"] == case.raw_id
        assert record["properties"]["source_file"] == f"{case.entity_type} 1.json"
        assert record["properties"]["demo_data"] is True
        assert len(result.meta.match_proofs) == 1
        proof = result.meta.match_proofs[0]
        assert proof.entity_id == case.stable_id
        assert proof.kind.value == expected_match_kind
        _assert_complete_evidence(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COMPANY_PARAMS)
async def test_every_company_location_is_returned_from_its_raw_headquarters_row(
    registry: ToolRegistry,
    case: EntityCase,
) -> None:
    matching = [
        relation
        for relation in RAW_ORACLE.relation_cases
        if relation.source_id == case.stable_id
        and relation.raw_relation == "Headquartered_in"
    ]
    assert len(matching) == 1
    headquarters = matching[0]
    assert headquarters.raw_tail == str(case.raw["city"])

    result = await registry.companies(
        {
            "query": case.raw_name,
            "match_mode": "exact",
            "include_headquarters": True,
        }
    )
    assert result.success is True
    records = _record_map(result)
    assert set(records) == {
        ("entity", case.stable_id),
        ("entity", headquarters.target_id),
        ("relation", headquarters.record_id),
    }
    location = records[("entity", headquarters.target_id)]
    assert location["entity_type"] == NodeType.LOCATION.value
    assert location["label"] == headquarters.raw_tail
    edge = records[("relation", headquarters.record_id)]
    assert edge["source"] == case.stable_id
    assert edge["target"] == headquarters.target_id
    assert edge["properties"]["raw_relation"] == "Headquartered_in"
    assert edge["properties"]["source_row"] == headquarters.row_number
    _assert_complete_evidence(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RELATION_PARAMS)
async def test_every_raw_relation_row_is_queryable_with_lossless_provenance(
    registry: ToolRegistry,
    case: RelationCase,
) -> None:
    result = await registry.relations(
        {
            "subject_ids": [case.source_id],
            "object_ids": [case.target_id],
            "direction": "outgoing",
            "relation_types": [],
            "raw_relation_types": [case.raw_relation],
            "include_endpoints": True,
            "limit": 200,
        }
    )
    assert result.success is True
    assert result.meta.truncated is False

    expected_group = RAW_ORACLE.relation_groups()[
        (case.source_id, case.raw_relation, case.target_id)
    ]
    expected_edge_ids = {item.record_id for item in expected_group}
    records = _record_map(result)
    returned_edge_ids = {
        record_id
        for (kind, record_id), _record in records.items()
        if kind == "relation"
    }
    assert returned_edge_ids == expected_edge_ids
    assert result.meta.total == result.meta.returned == len(expected_edge_ids)
    assert {
        record_id for (kind, record_id) in records if kind == "entity"
    } == {case.source_id, case.target_id}

    edge = records[("relation", case.record_id)]
    assert edge["source"] == case.source_id
    assert edge["target"] == case.target_id
    assert edge["label"] == case.raw_relation
    expected_properties = {
        "raw_head": case.raw_head,
        "raw_relation": case.raw_relation,
        "raw_tail": case.raw_tail,
        "source_file": "relations 1.json",
        "source_row": case.row_number,
        "demo_data": True,
    }
    assert {
        key: edge["properties"].get(key) for key in expected_properties
    } == expected_properties
    evidence = {item.id: item for item in result.evidence}
    assert case.evidence_id in edge["evidence_ids"]
    assert evidence[case.evidence_id].record_id == (
        f"relations 1.json#{case.row_number}"
    )
    assert evidence[case.evidence_id].source_kind == "raw_relation"
    _assert_complete_evidence(result)


def test_projection_exactly_retains_raw_duplicates_self_edges_and_references(
    repository: FixtureRepository,
) -> None:
    projected_ids = {edge.id for edge in repository.relations}
    assert projected_ids == {case.record_id for case in RAW_ORACLE.relation_cases}

    duplicate_groups = {
        key: cases
        for key, cases in RAW_ORACLE.relation_groups().items()
        if len(cases) > 1
    }
    assert len(duplicate_groups) == 4
    for (source_id, raw_relation, target_id), cases in duplicate_groups.items():
        projected = {
            edge.id
            for edge in repository.relations
            if edge.source == source_id
            and edge.target == target_id
            and edge.properties["raw_relation"] == raw_relation
        }
        assert projected == {case.record_id for case in cases}

    expected_self_edges = {
        case.record_id
        for case in RAW_ORACLE.relation_cases
        if case.source_id == case.target_id
    }
    assert len(expected_self_edges) == 2
    assert {
        edge.id for edge in repository.relations if edge.source == edge.target
    } == expected_self_edges

    expected_references = RAW_ORACLE.raw_reference_nodes()
    assert len(expected_references) == 9
    projected_references = {
        node.id: node
        for node in repository.nodes
        if node.properties.get("raw_reference_only") is True
    }
    assert set(projected_references) == set(expected_references)
    for node_id, (entity_type, label, first_row) in expected_references.items():
        node = projected_references[node_id]
        assert node.type.value == entity_type
        assert node.label == label
        expected_properties = {
            "source_id": label,
            "source_file": "relations 1.json",
            "source_row": first_row,
            "raw_reference_only": True,
            "demo_data": True,
        }
        assert {
            key: node.properties.get(key) for key in expected_properties
        } == expected_properties
