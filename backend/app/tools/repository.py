"""Validated, read-only projections over the three supplied raw JSON files.

The JSON files in the project's ``data/`` directory are the sole mock-data authority.  This module
does not read generated fixtures or curated facts.  It only projects raw records
into the typed graph contracts required by the API and keeps the original row,
relation vocabulary, source file, and line index on every projected edge.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.ids import normalize_query, slugify, stable_hash
from app.schemas import Evidence, GraphEdge, GraphNode, GraphPayload, NodeType, RelationType
from app.tools.contracts import (
    ENTITY_MATCH_ALGORITHM_VERSION,
    FUZZY_ACCEPT_THRESHOLD,
    FUZZY_MIN_MARGIN,
    EntityMatchProof,
    MatchKind,
    MatchMode,
    RawRelationType,
    RelationDirection,
)


RAW_FILES = (
    "person 1.json",
    "company 1.json",
    "relations 1.json",
)
PROVIDER = "local-raw-json-mock"
UNKNOWN_SOURCE_TIME = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class EntitySearchPage:
    nodes: tuple[GraphNode, ...]
    match_proofs: tuple[EntityMatchProof, ...]
    total: int
    truncated: bool
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class RelationSearchPage:
    edges: tuple[GraphEdge, ...]
    total: int
    truncated: bool


# RelationType is the API/query vocabulary.  The exact source vocabulary is never
# discarded: it remains the edge label and in ``properties.raw_relation``.
RAW_RELATION_TYPES: dict[str, tuple[RelationType, dict[str, Any]]] = {
    "CEO_of": (RelationType.WORKS_AT, {"role": "CEO", "status": "current"}),
    "Chairman_of": (
        RelationType.WORKS_AT,
        {"role": "chairman", "status": "current"},
    ),
    "Chairwoman_of": (
        RelationType.WORKS_AT,
        {"role": "chairwoman", "status": "current"},
    ),
    "Former_CEO_of": (
        RelationType.WORKS_AT,
        {"role": "CEO", "status": "former"},
    ),
    "Former_Chairman_of": (
        RelationType.WORKS_AT,
        {"role": "chairman", "status": "former"},
    ),
    "Former_President_of": (
        RelationType.WORKS_AT,
        {"role": "president", "status": "former"},
    ),
    "Founder_of": (RelationType.FOUNDED, {"founder_kind": "founder"}),
    "Co-founder_of": (RelationType.FOUNDED, {"founder_kind": "co-founder"}),
    "Headquartered_in": (RelationType.HEADQUARTERED_IN, {}),
    "Owns": (RelationType.OWNS, {}),
    "Partner_with": (RelationType.PARTNER_OF, {}),
    "Supplier_to": (RelationType.SUPPLIER_TO, {}),
    "Invested_in": (RelationType.INVESTED_IN, {}),
    "Competes_with": (
        RelationType.RELATED_TO,
        {"relation_kind": "competes_with"},
    ),
    "Uses_AI_from": (
        RelationType.RELATED_TO,
        {"relation_kind": "uses_ai_from"},
    ),
}


class DataValidationError(RuntimeError):
    """Raised when the checked-in demo fixture set is inconsistent."""


_LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_CORPORATE_SUFFIX_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "limited",
    "llc",
    "ltd",
}


def _latin_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _LATIN_TOKEN_PATTERN.findall(normalized)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_character in enumerate(right, start=1):
        current = [row]
        for column, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(left, right) / denominator)


def _soundex(value: str) -> str:
    """Small deterministic phonetic signal for Latin typo recovery only."""

    letters = "".join(character for character in value.upper() if "A" <= character <= "Z")
    if not letters:
        return ""
    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        "L": "4",
        **dict.fromkeys("MN", "5"),
        "R": "6",
    }
    result = [letters[0]]
    previous = codes.get(letters[0], "")
    for character in letters[1:]:
        code = codes.get(character, "")
        if code and code != previous:
            result.append(code)
        previous = code
    return ("".join(result) + "000")[:4]


def _fuzzy_score(query: str, label: str) -> tuple[float, str] | None:
    query_tokens = _latin_tokens(query)
    label_tokens = _latin_tokens(label)
    if not query_tokens or not label_tokens:
        # Deliberately no cross-script transliteration or model-knowledge aliases.
        return None

    # Corporate suffixes are weak identifiers and otherwise make every misspelled
    # "... Inc" query an ambiguous perfect match across unrelated companies.
    filtered_query = [
        token for token in query_tokens if token not in _CORPORATE_SUFFIX_TOKENS
    ]
    filtered_label = [
        token for token in label_tokens if token not in _CORPORATE_SUFFIX_TOKENS
    ]
    if filtered_query and filtered_label:
        query_tokens = filtered_query
        label_tokens = filtered_label

    candidates: list[tuple[float, str, str]] = []
    for query_token in query_tokens:
        for label_token in label_tokens:
            candidates.append(
                (_similarity(query_token, label_token), query_token, label_token)
            )

    if len(query_tokens) >= len(label_tokens):
        for start in range(len(query_tokens) - len(label_tokens) + 1):
            query_window = " ".join(
                query_tokens[start : start + len(label_tokens)]
            )
            label_value = " ".join(label_tokens)
            candidates.append(
                (_similarity(query_window, label_value), query_window, label_value)
            )

    lexical, query_value, matched_text = max(
        candidates,
        key=lambda item: (item[0], len(item[2]), item[2]),
    )
    query_phonetic = _soundex(query_value.replace(" ", ""))
    label_phonetic = _soundex(matched_text.replace(" ", ""))
    # A phonetic match is supporting evidence, never a perfect match.  0.84 was
    # chosen so Mask/Musk clears the 0.75 threshold and remains >0.08 above
    # Mask/Mark, whose Soundex codes differ.
    phonetic = 0.84 if query_phonetic and query_phonetic == label_phonetic else 0.0
    return max(lexical, phonetic), matched_text


def _fuzzy_match_proof(query: str, node: GraphNode) -> EntityMatchProof | None:
    scored = _fuzzy_score(query, node.label)
    if scored is None:
        return None
    score, matched_text = scored
    return EntityMatchProof(
        entity_id=node.id,
        query=query,
        matched_text=matched_text,
        kind=MatchKind.FUZZY,
        score=round(score, 6),
        algorithm=ENTITY_MATCH_ALGORITHM_VERSION,
    )


def _matches_subject_object(
    edge: GraphEdge,
    *,
    has_subject_filter: bool,
    subjects: set[str],
    has_object_filter: bool,
    objects: set[str],
    direction: RelationDirection,
) -> bool:
    if not has_subject_filter and not has_object_filter:
        return True
    if direction is RelationDirection.OUTGOING:
        return (not has_subject_filter or edge.source in subjects) and (
            not has_object_filter or edge.target in objects
        )
    if direction is RelationDirection.INCOMING:
        return (not has_subject_filter or edge.target in subjects) and (
            not has_object_filter or edge.source in objects
        )
    if has_subject_filter and has_object_filter:
        return (edge.source in subjects and edge.target in objects) or (
            edge.target in subjects and edge.source in objects
        )
    selected = subjects if has_subject_filter else objects
    return bool({edge.source, edge.target}.intersection(selected))


class AliasRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    normalized: str
    entity_id: str


class RawPerson(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    nationality: str
    summary: str


class RawCompany(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    legal_rep_id: str
    city: str
    founded_year: int


class RawRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    head: str
    relation: str
    tail: str


class FixtureRepository:
    """Immutable-in-practice fixture repository shared by all tool calls.

    Loading eagerly validates raw schemas, graph endpoints, exact source-name
    aliases, evidence coverage, relation mappings, and content hashes. Tool calls
    can therefore remain small and deterministic.
    """

    def __init__(
        self,
        *,
        directory: Path,
        manifest: dict[str, Any],
        nodes: Iterable[GraphNode],
        relations: Iterable[GraphEdge],
        evidence: Iterable[Evidence],
        aliases: Iterable[AliasRecord],
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.provider = str(manifest["provider"])
        self.data_version = str(manifest["data_version"])
        self.nodes_by_id = {node.id: node for node in nodes}
        self.relations_by_id = {edge.id: edge for edge in relations}
        self.evidence_by_id = {item.id: item for item in evidence}
        self.alias_records = tuple(aliases)
        self.alias_index = {item.normalized: item.entity_id for item in self.alias_records}
        # Public mapping used internally by the mock search implementation.
        self.aliases = dict(self.alias_index)
        self.source_id_index = {
            str(node.properties["source_id"]): node.id
            for node in self.nodes_by_id.values()
            if node.properties.get("source_id") is not None
        }
        self.is_loaded = True

    @classmethod
    def load(cls, directory: str | Path) -> FixtureRepository:
        requested_path = Path(directory).resolve()
        path = cls._find_raw_directory(requested_path)
        try:
            raw_payload = {
                file_name: json.loads((path / file_name).read_text(encoding="utf-8"))
                for file_name in RAW_FILES
            }
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"unable to load raw mock data from {path}: {exc}") from exc

        try:
            people_raw = cls._validate_raw_list(
                raw_payload["person 1.json"], RawPerson, "person 1.json"
            )
            companies_raw = cls._validate_raw_list(
                raw_payload["company 1.json"], RawCompany, "company 1.json"
            )
            relations_raw = cls._validate_raw_list(
                raw_payload["relations 1.json"], RawRelation, "relations 1.json"
            )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid raw mock-data schema: {exc}") from exc

        projected = cls._project_raw_records(people_raw, companies_raw, relations_raw)
        content_hash = stable_hash(raw_payload)
        manifest = {
            "dataset": "enterprise-relationship-public-raw-mock",
            "provider": PROVIDER,
            "schema_version": 1,
            "data_version": f"raw-v1-{content_hash[:16]}",
            "content_hash": content_hash,
            "is_demo": True,
            "source_files": list(RAW_FILES),
            "source_counts": {
                "persons": len(people_raw),
                "companies": len(companies_raw),
                "relations": len(relations_raw),
            },
            "output_counts": {
                "persons": len(projected["persons"]),
                "companies": len(projected["companies"]),
                "locations": len(projected["locations"]),
                "relations": len(projected["relations"]),
                "evidence": len(projected["evidence"]),
                "aliases": len(projected["aliases"]),
            },
        }

        repository = cls(
            directory=path,
            manifest=manifest,
            nodes=[
                *projected["persons"],
                *projected["companies"],
                *projected["locations"],
            ],
            relations=projected["relations"],
            evidence=projected["evidence"],
            aliases=projected["aliases"],
        )
        repository._validate(raw_payload, projected)
        return repository

    @staticmethod
    def _find_raw_directory(requested_path: Path) -> Path:
        if all((requested_path / file_name).is_file() for file_name in RAW_FILES):
            return requested_path
        expected = ", ".join(RAW_FILES)
        raise DataValidationError(
            f"raw mock-data files not found from {requested_path}; expected {expected}"
        )

    @staticmethod
    def _validate_raw_list(
        value: Any,
        model: type[RawPerson] | type[RawCompany] | type[RawRelation],
        file_name: str,
    ) -> list[RawPerson] | list[RawCompany] | list[RawRelation]:
        if not isinstance(value, list):
            raise ValueError(f"{file_name} must contain a top-level JSON array")
        return [model.model_validate(item) for item in value]

    @classmethod
    def _project_raw_records(
        cls,
        people_raw: list[RawPerson],
        companies_raw: list[RawCompany],
        relations_raw: list[RawRelation],
    ) -> dict[str, list[Any]]:
        people: list[GraphNode] = []
        companies: list[GraphNode] = []
        locations: list[GraphNode] = []
        relations: list[GraphEdge] = []
        evidence: list[Evidence] = []
        aliases: list[AliasRecord] = []
        raw_to_id: dict[str, str] = {}
        name_to_id: dict[str, str] = {}

        def add_evidence(
            evidence_id: str,
            record_id: str,
            source_kind: str,
        ) -> None:
            evidence.append(
                Evidence(
                    id=evidence_id,
                    provider=PROVIDER,
                    record_id=record_id,
                    source_kind=source_kind,
                    updated_at=UNKNOWN_SOURCE_TIME,
                    retrieved_at=UNKNOWN_SOURCE_TIME,
                    is_demo=True,
                )
            )

        def index_node(node: GraphNode, raw_values: Iterable[str]) -> None:
            for raw_value in raw_values:
                normalized = normalize_query(raw_value)
                previous = name_to_id.get(normalized)
                if previous is not None and previous != node.id:
                    raise DataValidationError(
                        f"ambiguous raw entity name {raw_value!r}: {previous} and {node.id}"
                    )
                name_to_id[normalized] = node.id
                aliases.append(
                    AliasRecord(alias=raw_value, normalized=normalized, entity_id=node.id)
                )

        for raw in people_raw:
            entity_id = f"person:{raw.id}"
            evidence_id = f"evidence:raw:person:{raw.id}"
            node = GraphNode(
                id=entity_id,
                type=NodeType.PERSON,
                label=raw.name,
                properties={
                    "source_id": raw.id,
                    "aliases": [raw.name],
                    "nationality": raw.nationality,
                    "summary": raw.summary,
                    "source_file": "person 1.json",
                    "demo_data": True,
                },
                evidence_ids=[evidence_id],
            )
            people.append(node)
            raw_to_id[raw.id] = entity_id
            index_node(node, [raw.name])
            add_evidence(evidence_id, raw.id, "raw_person")

        for raw in companies_raw:
            entity_id = f"company:{raw.id}"
            evidence_id = f"evidence:raw:company:{raw.id}"
            node = GraphNode(
                id=entity_id,
                type=NodeType.COMPANY,
                label=raw.name,
                properties={
                    "source_id": raw.id,
                    "aliases": [raw.name],
                    "legal_rep_id": raw.legal_rep_id,
                    "city": raw.city,
                    # This is a stable projection of the same raw ``city`` value,
                    # not an additional fact source.  The previous implementation
                    # advertised ``location_id`` as a selectable attribute without
                    # ever placing it on company records.
                    "location_id": f"location:{slugify(raw.city)}",
                    "founded_year": raw.founded_year,
                    "source_file": "company 1.json",
                    "demo_data": True,
                },
                evidence_ids=[evidence_id],
            )
            companies.append(node)
            raw_to_id[raw.id] = entity_id
            index_node(node, [raw.name])
            add_evidence(evidence_id, raw.id, "raw_company")

        companies_by_city: dict[str, list[str]] = {}
        for raw in companies_raw:
            companies_by_city.setdefault(raw.city, []).append(raw.id)
        relation_cities = {
            raw.tail
            for raw in relations_raw
            if raw.relation == "Headquartered_in"
        }
        location_to_id: dict[str, str] = {}
        for city in sorted(set(companies_by_city) | relation_cities):
            entity_id = f"location:{slugify(city)}"
            evidence_id = f"evidence:raw:location:{slugify(city)}"
            record_ids = companies_by_city.get(city, [])
            source_kind = "raw_company_city" if record_ids else "raw_relation_location"
            record_id = ",".join(record_ids) if record_ids else city
            node = GraphNode(
                id=entity_id,
                type=NodeType.LOCATION,
                label=city,
                properties={
                    "source_id": city,
                    "aliases": [city],
                    "source_file": (
                        "company 1.json" if record_ids else "relations 1.json"
                    ),
                    "demo_data": True,
                },
                evidence_ids=[evidence_id],
            )
            locations.append(node)
            location_to_id[city] = entity_id
            index_node(node, [city])
            add_evidence(evidence_id, record_id, source_kind)

        reference_nodes: dict[tuple[NodeType, str], GraphNode] = {}

        def endpoint_id(raw_value: str, raw_relation: str, position: str, row: int) -> str:
            known = raw_to_id.get(raw_value) or name_to_id.get(normalize_query(raw_value))
            if known is not None:
                return known
            if raw_relation == "Headquartered_in" and position == "tail":
                return location_to_id[raw_value]

            person_side = raw_relation in {
                "CEO_of",
                "Chairman_of",
                "Chairwoman_of",
                "Former_CEO_of",
                "Former_Chairman_of",
                "Former_President_of",
                "Founder_of",
                "Co-founder_of",
            } and position == "head"
            node_type = NodeType.PERSON if person_side else NodeType.COMPANY
            key = (node_type, raw_value)
            if key not in reference_nodes:
                entity_id = f"{node_type.value}:raw-reference:{slugify(raw_value)}"
                evidence_id = (
                    f"evidence:raw:{node_type.value}-reference:{slugify(raw_value)}"
                )
                node = GraphNode(
                    id=entity_id,
                    type=node_type,
                    label=raw_value,
                    properties={
                        "source_id": raw_value,
                        "aliases": [raw_value],
                        "source_file": "relations 1.json",
                        "source_row": row,
                        "raw_reference_only": True,
                        "demo_data": True,
                    },
                    evidence_ids=[evidence_id],
                )
                reference_nodes[key] = node
                if node_type is NodeType.PERSON:
                    people.append(node)
                else:
                    companies.append(node)
                index_node(node, [raw_value])
                add_evidence(
                    evidence_id,
                    f"relations 1.json#{row}",
                    "raw_relation_reference",
                )
            return reference_nodes[key].id

        for row, raw in enumerate(relations_raw, start=1):
            mapping = RAW_RELATION_TYPES.get(raw.relation)
            if mapping is None:
                raise DataValidationError(
                    f"unsupported raw relation at relations 1.json#{row}: {raw.relation}"
                )
            relation_type, mapped_properties = mapping
            source = endpoint_id(raw.head, raw.relation, "head", row)
            target = endpoint_id(raw.tail, raw.relation, "tail", row)
            edge_id = f"relation:raw:{row:04d}"
            evidence_id = f"evidence:raw:relation:{row:04d}"
            relations.append(
                GraphEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    type=relation_type,
                    label=raw.relation,
                    properties={
                        "raw_head": raw.head,
                        "raw_relation": raw.relation,
                        "raw_tail": raw.tail,
                        "source_file": "relations 1.json",
                        "source_row": row,
                        "demo_data": True,
                        **mapped_properties,
                    },
                    evidence_ids=[evidence_id],
                )
            )
            add_evidence(
                evidence_id,
                f"relations 1.json#{row}",
                "raw_relation",
            )

        return {
            "persons": sorted(people, key=lambda item: item.id),
            "companies": sorted(companies, key=lambda item: item.id),
            "locations": sorted(locations, key=lambda item: item.id),
            "relations": relations,
            "evidence": sorted(evidence, key=lambda item: item.id),
            "aliases": sorted(
                aliases,
                key=lambda item: (item.normalized, item.entity_id),
            ),
        }

    def _validate(
        self,
        raw: dict[str, Any],
        projected: dict[str, list[Any]],
    ) -> None:
        expected_hash = stable_hash(raw)
        if self.manifest.get("content_hash") != expected_hash:
            raise DataValidationError("raw content hash does not match manifest")
        expected_version = f"raw-v1-{expected_hash[:16]}"
        if self.data_version != expected_version:
            raise DataValidationError("raw data_version does not match content hash")
        if self.manifest.get("is_demo") is not True:
            raise DataValidationError("raw manifest must be explicitly marked as demo data")
        if self.manifest.get("source_files") != list(RAW_FILES):
            raise DataValidationError("only the three supplied raw files may be data sources")

        persons = projected["persons"]
        companies = projected["companies"]
        locations = projected["locations"]
        all_nodes = [*persons, *companies, *locations]
        if len(self.nodes_by_id) != len(all_nodes):
            raise DataValidationError("duplicate projected entity IDs")
        if len(self.relations_by_id) != len(raw["relations 1.json"]):
            raise DataValidationError("every raw relation row must be projected exactly once")
        if len(self.evidence_by_id) != len(projected["evidence"]):
            raise DataValidationError("duplicate projected evidence IDs")

        expected_types = {
            NodeType.PERSON: persons,
            NodeType.COMPANY: companies,
            NodeType.LOCATION: locations,
        }
        for node_type, nodes in expected_types.items():
            if not nodes or any(node.type != node_type for node in nodes):
                raise DataValidationError(f"invalid or empty projected {node_type.value} set")
            if any(not node.id.startswith(f"{node_type.value}:") for node in nodes):
                raise DataValidationError(f"non-namespaced {node_type.value} ID")

        missing_endpoints = {
            endpoint
            for edge in self.relations_by_id.values()
            for endpoint in (edge.source, edge.target)
            if endpoint not in self.nodes_by_id
        }
        if missing_endpoints:
            raise DataValidationError(
                f"dangling projected relation endpoints: {sorted(missing_endpoints)}"
            )

        for record in [*all_nodes, *self.relations_by_id.values()]:
            if not record.evidence_ids:
                raise DataValidationError(f"record {record.id} has no evidence")
            missing = set(record.evidence_ids) - self.evidence_by_id.keys()
            if missing:
                raise DataValidationError(
                    f"record {record.id} references missing evidence: {sorted(missing)}"
                )
        if any(
            not item.is_demo or item.provider != self.provider
            for item in self.evidence_by_id.values()
        ):
            raise DataValidationError(
                "all projected evidence must be demo data from the raw-data provider"
            )

        seen_aliases: dict[str, str] = {}
        for alias in self.alias_records:
            if alias.entity_id not in self.nodes_by_id:
                raise DataValidationError(f"alias references unknown entity: {alias.entity_id}")
            if alias.normalized != normalize_query(alias.alias):
                raise DataValidationError(f"alias is not normalized: {alias.alias!r}")
            previous = seen_aliases.get(alias.normalized)
            if previous is not None and previous != alias.entity_id:
                raise DataValidationError(f"ambiguous alias: {alias.alias!r}")
            seen_aliases[alias.normalized] = alias.entity_id

        relation_types = {edge.type for edge in self.relations_by_id.values()}
        required_relation_types = {item[0] for item in RAW_RELATION_TYPES.values()}
        if not required_relation_types.issubset(relation_types):
            missing = required_relation_types - relation_types
            raise DataValidationError(f"missing required relation types: {sorted(missing)}")
        if RelationType.CONTROLS in relation_types:
            raise DataValidationError("raw files contain no explicit controls relation")
        for row, edge in enumerate(self.relations_by_id.values(), start=1):
            if edge.properties.get("source_file") != "relations 1.json":
                raise DataValidationError(f"relation {edge.id} lost its raw source file")
            if not edge.properties.get("raw_relation"):
                raise DataValidationError(f"relation {edge.id} lost its raw vocabulary")

    def assert_ready(self) -> None:
        if not self.is_loaded or not self.nodes_by_id or not self.relations_by_id:
            raise DataValidationError("fixture repository is not ready")

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(self.nodes_by_id[key] for key in sorted(self.nodes_by_id))

    @property
    def relations(self) -> tuple[GraphEdge, ...]:
        return tuple(self.relations_by_id[key] for key in sorted(self.relations_by_id))

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self.evidence_by_id[key] for key in sorted(self.evidence_by_id))

    def compact_planner_catalog(self) -> dict[str, Any]:
        """Return the small, fact-free vocabulary Planner may use for alignment.

        The catalog is derived on every application startup from the same projected
        raw records used by the mock tools.  It intentionally exposes no stable ID,
        entity property, relation endpoint, or inferred alias: Planner may select a
        raw name/type and a raw relation word, while Researcher must still call the
        tools to verify every entity and relationship.
        """

        source_file_by_type = {
            NodeType.PERSON: "person 1.json",
            NodeType.COMPANY: "company 1.json",
        }
        order = {NodeType.PERSON: 0, NodeType.COMPANY: 1}
        entities = [
            {"name": node.label, "entity_type": node.type.value}
            for node in sorted(
                (
                    item
                    for item in self.nodes_by_id.values()
                    if item.type in source_file_by_type
                    and item.properties.get("source_file")
                    == source_file_by_type[item.type]
                ),
                key=lambda item: (order[item.type], item.label.casefold()),
            )
        ]
        raw_relation_vocabulary = sorted(
            {
                str(edge.properties["raw_relation"])
                for edge in self.relations_by_id.values()
                if edge.properties.get("raw_relation")
            },
            key=str.casefold,
        )
        return {
            "entity_catalog": entities,
            "raw_relation_vocabulary": raw_relation_vocabulary,
        }

    def canonical_entity_id(self, value: str) -> str | None:
        if value in self.nodes_by_id:
            return value
        if value in self.source_id_index:
            return self.source_id_index[value]
        return self.alias_index.get(normalize_query(value))

    def resolve_alias(self, value: str, node_type: NodeType | None = None) -> str | None:
        entity_id = self.canonical_entity_id(value)
        if entity_id is None:
            return None
        node = self.nodes_by_id[entity_id]
        return entity_id if node_type is None or node.type == node_type else None

    def find_mentions(
        self,
        text: str,
        node_types: Iterable[NodeType] | None = None,
    ) -> list[str]:
        normalized_text = normalize_query(text)
        allowed = set(node_types) if node_types is not None else set(NodeType)
        matches: set[str] = set()
        for alias, entity_id in self.alias_index.items():
            if len(alias) >= 2 and alias in normalized_text:
                if self.nodes_by_id[entity_id].type in allowed:
                    matches.add(entity_id)
        return sorted(matches)

    def search_entities(
        self,
        *,
        node_type: NodeType,
        query: str | None = None,
        entity_ids: Iterable[str] = (),
        limit: int = 100,
    ) -> list[GraphNode]:
        """Compatibility wrapper for exact entity lookup."""

        return list(
            self.search_entities_page(
                node_type=node_type,
                query=query,
                entity_ids=entity_ids,
                match_mode=MatchMode.EXACT,
                limit=limit,
            ).nodes
        )

    def search_entities_page(
        self,
        *,
        node_type: NodeType,
        query: str | None = None,
        entity_ids: Iterable[str] = (),
        match_mode: MatchMode = MatchMode.EXACT,
        limit: int = 100,
    ) -> EntitySearchPage:
        """Return exact or explicitly requested fuzzy candidates with proof.

        Fuzzy lookup never adds aliases.  It compares the supplied Latin text only
        with names already present in the three raw JSON files. Exact matches always
        win, even when ``match_mode=fuzzy``.
        """

        has_id_filter, requested_ids = self._canonical_id_filter(entity_ids)
        if has_id_filter and not requested_ids:
            return EntitySearchPage((), (), 0, False)
        candidates = {
            node.id: node
            for node in self.nodes_by_id.values()
            if node.type == node_type
            and (not has_id_filter or node.id in requested_ids)
        }
        if not query:
            ordered = [candidates[key] for key in sorted(candidates)]
            selected = ordered[:limit]
            proofs = tuple(
                EntityMatchProof(
                    entity_id=node.id,
                    query=node.id,
                    matched_text=node.label,
                    kind=MatchKind.EXPLICIT_ID,
                    score=1.0,
                )
                for node in selected
            )
            return EntitySearchPage(
                tuple(selected), proofs, len(ordered), len(selected) < len(ordered)
            )

        exact_proofs = self._exact_entity_match_proofs(query, candidates.values())
        if exact_proofs:
            ordered_proofs = tuple(
                sorted(exact_proofs, key=lambda item: (item.entity_id, item.kind.value))
            )
            ordered_nodes = [candidates[item.entity_id] for item in ordered_proofs]
            selected_nodes = ordered_nodes[:limit]
            selected_ids = {node.id for node in selected_nodes}
            selected_proofs = tuple(
                proof for proof in ordered_proofs if proof.entity_id in selected_ids
            )
            return EntitySearchPage(
                tuple(selected_nodes),
                selected_proofs,
                len(ordered_nodes),
                len(selected_nodes) < len(ordered_nodes),
            )

        if match_mode is MatchMode.EXACT:
            return EntitySearchPage((), (), 0, False)

        fuzzy_proofs = sorted(
            (
                proof
                for node in candidates.values()
                if (proof := _fuzzy_match_proof(query, node)) is not None
            ),
            key=lambda item: (-item.score, item.entity_id),
        )
        qualifying = [
            proof for proof in fuzzy_proofs if proof.score >= FUZZY_ACCEPT_THRESHOLD
        ]
        if not qualifying:
            # The best considered candidates remain visible as proof of why no raw
            # record was accepted, but no entity record crosses the fact boundary.
            return EntitySearchPage((), tuple(fuzzy_proofs[:3]), 0, False)

        ambiguous = (
            len(qualifying) > 1
            and qualifying[0].score - qualifying[1].score < FUZZY_MIN_MARGIN
        )
        accepted = qualifying if ambiguous else qualifying[:1]
        selected = accepted[:limit]
        selected_nodes = tuple(candidates[proof.entity_id] for proof in selected)
        # Include the runner-up so the acceptance margin is independently auditable.
        proof_count = max(len(selected), 2)
        return EntitySearchPage(
            selected_nodes,
            tuple(fuzzy_proofs[:proof_count]),
            len(accepted),
            len(selected) < len(accepted),
            ambiguous=ambiguous,
        )

    def _exact_entity_match_proofs(
        self,
        query: str,
        candidates: Iterable[GraphNode],
    ) -> list[EntityMatchProof]:
        normalized_query = normalize_query(query)
        matches: dict[str, EntityMatchProof] = {}
        priority = {
            MatchKind.EXPLICIT_ID: 4,
            MatchKind.EXACT_NAME: 3,
            MatchKind.MENTION: 2,
            MatchKind.SUBSTRING: 1,
        }
        for node in candidates:
            raw_values = [
                str(node.properties.get("source_id", "")),
                node.id,
                node.label,
                *[str(item) for item in node.properties.get("aliases", [])],
            ]
            best: EntityMatchProof | None = None
            for value in dict.fromkeys(item for item in raw_values if item):
                normalized_value = normalize_query(value)
                if normalized_query == normalized_value:
                    kind = (
                        MatchKind.EXPLICIT_ID
                        if value in {node.id, str(node.properties.get("source_id", ""))}
                        else MatchKind.EXACT_NAME
                    )
                elif len(normalized_value) >= 2 and normalized_value in normalized_query:
                    kind = MatchKind.MENTION
                elif normalized_query and normalized_query in normalized_value:
                    kind = MatchKind.SUBSTRING
                else:
                    continue
                proof = EntityMatchProof(
                    entity_id=node.id,
                    query=query,
                    matched_text=value,
                    kind=kind,
                    score=1.0,
                )
                if best is None or priority[kind] > priority[best.kind]:
                    best = proof
            if best is not None:
                matches[node.id] = best
        return list(matches.values())

    def query_relations(
        self,
        *,
        subject_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        relation_types: Iterable[RelationType] = (),
        raw_relation_types: Iterable[RawRelationType | str] = (),
        direction: RelationDirection = RelationDirection.ANY,
        limit: int = 200,
    ) -> RelationSearchPage:
        direction = RelationDirection(direction)
        has_subject_filter, subjects = self._canonical_id_filter(subject_ids)
        has_object_filter, objects = self._canonical_id_filter(object_ids)
        if (
            (has_subject_filter and not subjects)
            or (has_object_filter and not objects)
        ):
            return RelationSearchPage((), 0, False)
        types = set(relation_types)
        raw_types = {
            item.value if isinstance(item, RawRelationType) else str(item)
            for item in raw_relation_types
        }
        matches: list[GraphEdge] = []
        for edge in self.relations:
            if types and edge.type not in types:
                continue
            if raw_types and str(edge.properties.get("raw_relation")) not in raw_types:
                continue
            if not _matches_subject_object(
                edge,
                has_subject_filter=has_subject_filter,
                subjects=subjects,
                has_object_filter=has_object_filter,
                objects=objects,
                direction=direction,
            ):
                continue
            matches.append(edge)
        selected = matches[:limit]
        return RelationSearchPage(
            tuple(selected), len(matches), len(selected) < len(matches)
        )

    def _canonical_id_filter(self, values: Iterable[str]) -> tuple[bool, set[str]]:
        requested = tuple(values)
        canonical_ids = {
            canonical
            for item in requested
            if (canonical := self.canonical_entity_id(item)) is not None
        }
        return bool(requested), canonical_ids

    def node_record(
        self,
        node: GraphNode,
        attributes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        properties = dict(node.properties)
        if attributes is not None:
            # Attribute selection controls user-facing business fields, not source
            # provenance.  Entity records reached through a name lookup and the
            # same entities reached as relation endpoints must project identical
            # immutable source identity; otherwise a verified zero-result graph can
            # lose ``source_file`` while a non-empty relation graph retains it.
            allowed = {
                *attributes,
                "source_id",
                "source_file",
                "demo_data",
            }
            properties = {key: value for key, value in properties.items() if key in allowed}
        return {
            "record_kind": "entity",
            "id": node.id,
            "entity_type": node.type.value,
            "label": node.label,
            "properties": properties,
            "evidence_ids": list(node.evidence_ids),
        }

    def relation_record(self, edge: GraphEdge) -> dict[str, Any]:
        return {
            "record_kind": "relation",
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation_type": edge.type.value,
            "label": edge.label,
            "properties": dict(edge.properties),
            "evidence_ids": list(edge.evidence_ids),
        }

    def evidence_for_records(self, records: Iterable[dict[str, Any]]) -> list[Evidence]:
        ids = {
            evidence_id
            for record in records
            for evidence_id in record.get("evidence_ids", [])
        }
        return [self.evidence_by_id[item] for item in sorted(ids)]

    def session_graph_is_trusted(self, value: GraphPayload | dict[str, Any]) -> bool:
        """Verify that restored session facts are a projection of current fixtures.

        Checkpoints are durable input, so schema validity and a matching data-version
        string alone are insufficient.  A trusted graph must have an intact content
        hash and every entity, relation, property, and provenance record must be a
        subset of the currently loaded immutable mock dataset.
        """

        from app.memory.graph_ops import graph_id_for

        try:
            graph = GraphPayload.model_validate(value)
        except (TypeError, ValueError):
            return False
        if graph.data_version != self.data_version:
            return False
        if graph.graph_id != graph_id_for(
            graph.nodes, graph.edges, graph.data_version, graph.evidence
        ):
            return False
        if len({node.id for node in graph.nodes}) != len(graph.nodes):
            return False
        if len({edge.id for edge in graph.edges}) != len(graph.edges):
            return False

        for node in graph.nodes:
            fixture = self.nodes_by_id.get(node.id)
            if fixture is None or (node.type, node.label) != (fixture.type, fixture.label):
                return False
            if set(node.evidence_ids) != set(fixture.evidence_ids):
                return False
            if any(fixture.properties.get(key) != item for key, item in node.properties.items()):
                return False

        for edge in graph.edges:
            fixture = self.relations_by_id.get(edge.id)
            if fixture is None:
                return False
            if (
                edge.source,
                edge.target,
                edge.type,
                edge.label,
            ) != (
                fixture.source,
                fixture.target,
                fixture.type,
                fixture.label,
            ):
                return False
            if set(edge.evidence_ids) != set(fixture.evidence_ids):
                return False
            if any(fixture.properties.get(key) != item for key, item in edge.properties.items()):
                return False

        for item in graph.evidence:
            fixture = self.evidence_by_id.get(item.id)
            if fixture is None:
                return False
            if item.model_dump(exclude={"retrieved_at"}) != fixture.model_dump(
                exclude={"retrieved_at"}
            ):
                return False
        return True
