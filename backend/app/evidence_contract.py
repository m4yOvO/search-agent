"""Deterministic evidence checks shared by prompt-driven agent roles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.schemas import Intent, NodeType, QuerySignature, RelationType, ResultMergeStrategy


RELATIONAL_INTENTS = frozenset(
    {
        Intent.FIND_CONTROLLED_COMPANIES,
        Intent.FIND_RELATED_COMPANIES,
        Intent.LOCATE_ENTITIES,
    }
)


def requires_explicit_relations(signature: QuerySignature) -> bool:
    """Return whether a successful result must contain explicit relation records."""

    return signature.intent in RELATIONAL_INTENTS or bool(signature.relation_types)


def validate_signature_records(
    signature: QuerySignature,
    selected_records: Iterable[Mapping[str, Any]],
    all_records: Iterable[Mapping[str, Any]],
) -> None:
    """Prove that a canonical signature is supported by selected mock records.

    A blank relation filter is the Planner's explicit request for the complete direct
    relation scope.  It is therefore not an absent task and must not be replaced with
    an intent-specific list.  Non-empty filters remain strict allow-lists.  Every
    selected edge and endpoint must still come from verified tool records.
    """

    selected = list(selected_records)
    catalog = list(all_records)
    entities = {
        str(record["id"]): record
        for record in catalog
        if record.get("record_kind") == "entity" and record.get("id")
    }
    relations = [
        record
        for record in selected
        if record.get("record_kind") == "relation" and record.get("id")
    ]

    relation_required = requires_explicit_relations(signature)
    if relation_required and not signature.subject_ids:
        raise ValueError("a relational signature requires verified subject IDs")
    if relation_required and not relations:
        raise ValueError("a relational result requires selected relation records")
    if not relations:
        if signature.intent in {Intent.CLARIFY, Intent.UNSUPPORTED}:
            raise ValueError("non-research intents cannot produce a factual result")
        selected_entities = {
            str(record["id"]): record
            for record in selected
            if record.get("record_kind") == "entity" and record.get("id")
        }
        signed_ids = {*signature.subject_ids, *signature.object_ids}
        if not signed_ids:
            raise ValueError("a factual profile signature requires verified entity IDs")
        if set(selected_entities) != signed_ids:
            raise ValueError("profile signature IDs must exactly match selected entities")

        expected_profile_type = {
            Intent.GET_COMPANY_PROFILE: NodeType.COMPANY.value,
            Intent.GET_PERSON_PROFILE: NodeType.PERSON.value,
        }.get(signature.intent)
        if expected_profile_type is None:
            raise ValueError("a non-relational factual intent is unsupported")
        if any(
            str(record.get("entity_type", "")) != expected_profile_type
            for record in selected_entities.values()
        ):
            raise ValueError("selected profile entity type does not match the intent")
        requested_target_types = {item.value for item in signature.target_types}
        if requested_target_types and requested_target_types != {expected_profile_type}:
            raise ValueError("profile target type does not match the selected entity type")
        for record in selected_entities.values():
            properties = record.get("properties") or {}
            missing_attributes = set(signature.requested_attributes) - set(properties)
            if missing_attributes:
                raise ValueError("requested profile attributes are absent from selected records")
        return

    allowed_types = set(signature.relation_types)
    allowed_raw_relations = set(signature.raw_relation_qualifiers)
    selected_types: set[RelationType] = set()
    endpoints: set[str] = set()
    endpoint_types: set[str] = set()
    endpoint_pairs: list[tuple[str, str]] = []
    for record in relations:
        relation_type = RelationType(str(record.get("relation_type")))
        if allowed_types and relation_type not in allowed_types:
            raise ValueError("selected relation type is outside the canonical query scope")
        raw_relation = str((record.get("properties") or {}).get("raw_relation", ""))
        if allowed_raw_relations and raw_relation not in allowed_raw_relations:
            raise ValueError("selected raw relation is outside the canonical qualifier scope")
        source = str(record.get("source", ""))
        target = str(record.get("target", ""))
        if not source or not target or source not in entities or target not in entities:
            raise ValueError("selected relation endpoints must be verified entity records")
        selected_types.add(relation_type)
        endpoints.update((source, target))
        endpoint_pairs.append((source, target))
        endpoint_types.update(
            (
                str(entities[source].get("entity_type", "")),
                str(entities[target].get("entity_type", "")),
            )
        )

    if not selected_types:
        raise ValueError("selected relations did not provide a valid relation type")
    missing_subjects = set(signature.subject_ids) - endpoints
    if (
        missing_subjects
        and signature.result_merge is not ResultMergeStrategy.UNION
    ):
        raise ValueError("signature subjects are not endpoints of selected relations")
    if set(signature.object_ids) - endpoints:
        raise ValueError("signature objects are not endpoints of selected relations")
    if signature.subject_ids and signature.object_ids:
        subjects = set(signature.subject_ids)
        objects = set(signature.object_ids)
        if not any(
            (source in subjects and target in objects)
            or (source in objects and target in subjects)
            for source, target in endpoint_pairs
        ):
            raise ValueError("signature subjects and objects have no selected relation")

    # Result merging is represented on the existing canonical boundary by the
    # association operator derived from Planner.result_merge.  UNION permits one
    # seed to contribute an exhaustive empty set.  INTERSECTION requires every
    # signed result object to be connected to every signed subject.  DIRECT requires
    # each selected edge to connect signed endpoints instead of introducing a third
    # neighbour.  These are evidence checks, not query interpretation.
    if signature.result_merge is ResultMergeStrategy.INTERSECTION:
        subjects = set(signature.subject_ids)
        for object_id in signature.object_ids:
            connected_subjects = {
                source if target == object_id else target
                for source, target in endpoint_pairs
                if object_id in {source, target}
                and (source in subjects or target in subjects)
            }
            if connected_subjects != subjects:
                raise ValueError(
                    "intersection result objects must connect to every signed subject"
                )
    elif signature.result_merge is ResultMergeStrategy.DIRECT:
        signed = {*signature.subject_ids, *signature.object_ids}
        if any(source not in signed or target not in signed for source, target in endpoint_pairs):
            raise ValueError("direct merge contains an unsigned intermediate endpoint")
    requested_target_types = {item.value for item in signature.target_types}
    if requested_target_types - endpoint_types:
        raise ValueError("signature target types are absent from selected relation endpoints")
    if requested_target_types and any(
        str(entities.get(object_id, {}).get("entity_type", ""))
        not in requested_target_types
        for object_id in signature.object_ids
    ):
        raise ValueError("signed result objects fall outside the requested target types")


def expected_focus_entity_ids(
    signature: QuerySignature,
    selected_records: Iterable[Mapping[str, Any]],
    all_records: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Derive the complete follow-up referent set from verified semantics.

    Focus is a redundant memory projection, not a new model-selected fact.  The
    Planner/Researcher still decide the intent, signature, and selected records;
    this function only prevents a provider from dropping arbitrary members of the
    already verified result set.
    """

    selected = list(selected_records)
    catalog = list(all_records)
    entities = {
        str(record["id"]): record
        for record in catalog
        if record.get("record_kind") == "entity" and record.get("id")
    }
    relations = [
        record for record in selected if record.get("record_kind") == "relation"
    ]
    selected_entities = {
        str(record["id"])
        for record in selected
        if record.get("record_kind") == "entity" and record.get("id")
    }

    if signature.intent is Intent.LOCATE_ENTITIES:
        expected = set(signature.subject_ids)
    elif relations:
        endpoints = {
            str(endpoint)
            for record in relations
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        target_types = {item.value for item in signature.target_types}
        if signature.result_merge is ResultMergeStrategy.UNION:
            expected = set(signature.object_ids)
        elif target_types:
            expected = {
                entity_id
                for entity_id in endpoints
                if entity_id not in set(signature.subject_ids)
                and str(entities.get(entity_id, {}).get("entity_type", ""))
                in target_types
            }
        else:
            expected = set(signature.object_ids) or (
                endpoints - set(signature.subject_ids)
            )
    else:
        expected = {
            *signature.subject_ids,
            *signature.object_ids,
        }

    eligible = selected_entities | {
        str(endpoint)
        for record in relations
        for endpoint in (record.get("source"), record.get("target"))
        if endpoint is not None
    }
    if not expected or expected - entities.keys() or expected - eligible:
        raise ValueError("verified records do not provide a complete focus set")
    return sorted(expected)
