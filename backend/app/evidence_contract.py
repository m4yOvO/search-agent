"""Deterministic evidence checks shared by prompt-driven agent roles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.schemas import (
    GoalResultStatus,
    Intent,
    NodeType,
    QueryGoalSignature,
    QuerySignature,
    RelationType,
    ResultMergeStrategy,
)


RELATIONAL_INTENTS = frozenset(
    {
        Intent.FIND_CONTROLLED_COMPANIES,
        Intent.FIND_RELATED_COMPANIES,
        Intent.LOCATE_ENTITIES,
    }
)


def requires_explicit_relations(signature: QuerySignature) -> bool:
    """Return whether a successful result must contain explicit relation records."""

    if signature.goals:
        return any(
            goal.intent in RELATIONAL_INTENTS or bool(goal.relation_types)
            for goal in signature.goals
        )
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
    if signature.goals:
        _validate_goal_signature_records(signature, selected, catalog)
        return
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
        and signature.result_merge
        not in {ResultMergeStrategy.UNION, ResultMergeStrategy.DIRECT}
        and signature.intent is not Intent.LOCATE_ENTITIES
    ):
        raise ValueError("signature subjects are not endpoints of selected relations")
    if (
        set(signature.object_ids) - endpoints
        and signature.result_merge is not ResultMergeStrategy.DIRECT
    ):
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

    if signature.goals:
        expected = {
            entity_id
            for goal in signature.goals
            for entity_id in goal.focus_entity_ids
        }
        eligible = selected_entities | {
            str(endpoint)
            for record in relations
            for endpoint in (record.get("source"), record.get("target"))
            if endpoint is not None
        }
        if not expected and all(
            goal.result_status is GoalResultStatus.SKIPPED_EMPTY_INPUT
            for goal in signature.goals
        ):
            return []
        if not expected or expected - entities.keys() or expected - eligible:
            raise ValueError("verified goal records do not provide a complete focus set")
        return sorted(expected)

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
        if signature.result_merge is ResultMergeStrategy.DIRECT:
            # A direct query returns the induced subgraph over its operands.  All
            # operands remain the natural referent set even when one has no edge.
            expected = {*signature.subject_ids, *signature.object_ids}
        elif signature.object_ids:
            # Researcher derives signed result objects from the verified relation
            # projection.  Prefer that role-aware set over reconstructing neighbours
            # by subtracting subjects from endpoints: for a retained self relation,
            # the same entity is both the subject and its legitimate result object.
            expected = set(signature.object_ids)
            if expected - entities.keys() or expected - endpoints:
                raise ValueError(
                    "verified records do not provide a complete focus set"
                )
            if target_types and any(
                str(entities.get(entity_id, {}).get("entity_type", ""))
                not in target_types
                for entity_id in expected
            ):
                raise ValueError(
                    "signed focus objects fall outside the requested target types"
                )
        elif target_types:
            # Compatibility path for signatures created before result objects were
            # signed explicitly.  Current v5 goal signatures take the branch above.
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


def _validate_goal_signature_records(
    signature: QuerySignature,
    selected: list[Mapping[str, Any]],
    catalog: list[Mapping[str, Any]],
) -> None:
    """Validate each goal against only its runtime-derived result record set."""

    selected_by_id = {
        str(record["id"]): record
        for record in selected
        if record.get("id") is not None
    }
    catalog_entities = {
        str(record["id"])
        for record in catalog
        if record.get("record_kind") == "entity" and record.get("id")
    }
    goal_by_id = {goal.goal_id: goal for goal in signature.goals}
    assigned_relation_ids: set[str] = set()
    for goal in signature.goals:
        missing_subjects = set(goal.subject_ids) - catalog_entities
        if missing_subjects:
            raise ValueError("goal signature subjects are not verified entities")
        if set(goal.context_entity_ids) - set(signature.context_entity_ids):
            raise ValueError("goal context IDs are absent from the query signature")
        if goal.result_status is GoalResultStatus.SKIPPED_EMPTY_INPUT:
            if not goal.depends_on_goal_ids:
                raise ValueError("an empty-input goal requires a verified dependency")
            if not any(
                goal_by_id[dependency].result_status
                is not GoalResultStatus.NONEMPTY
                for dependency in goal.depends_on_goal_ids
            ):
                raise ValueError("an empty-input goal requires an empty dependency")
            continue
        if goal.result_status is GoalResultStatus.VERIFIED_EMPTY:
            continue

        goal_records: list[Mapping[str, Any]] = []
        for record_id in goal.result_record_ids:
            record = selected_by_id.get(record_id)
            if record is None:
                raise ValueError("goal result record is absent from selected records")
            goal_records.append(record)
            if record.get("record_kind") == "relation":
                assigned_relation_ids.add(record_id)
        required_entity_proofs: set[str] = set()
        if goal.aggregation is ResultMergeStrategy.DIRECT:
            # A non-empty induced subgraph may contain verified operands with no
            # incident edge.  Those entities are still part of the direct result's
            # natural focus.
            required_entity_proofs.update(goal.subject_ids)
            required_entity_proofs.update(goal.object_ids)
        if goal.intent is Intent.LOCATE_ENTITIES:
            # A complete batch location query retains every queried company as
            # focus, including a company with no headquarters edge.
            required_entity_proofs.update(goal.subject_ids)
        if required_entity_proofs - set(goal.result_record_ids):
            raise ValueError(
                "goal result records must include every focus entity without edge proof"
            )
        legacy = _goal_as_query_signature(signature, goal)
        validate_signature_records(legacy, goal_records, catalog)

    selected_relation_ids = {
        str(record["id"])
        for record in selected
        if record.get("record_kind") == "relation" and record.get("id")
    }
    if selected_relation_ids != assigned_relation_ids:
        raise ValueError("selected relation records are not assigned to query goals")


def _goal_as_query_signature(
    parent: QuerySignature, goal: QueryGoalSignature
) -> QuerySignature:
    """Project one signed goal through the existing single-goal evidence checks."""

    return QuerySignature(
        version=parent.version,
        intent=goal.intent,
        subject_ids=goal.subject_ids,
        object_ids=goal.object_ids,
        relation_types=goal.effective_relation_types,
        requested_relation_types=goal.requested_relation_types,
        effective_relation_types=goal.effective_relation_types,
        raw_relation_qualifiers=goal.raw_relation_qualifiers,
        verified_empty_relation_types=goal.verified_empty_relation_types,
        target_types=goal.target_types,
        requested_attributes=goal.requested_attributes,
        context_entity_ids=goal.context_entity_ids,
        result_merge=goal.aggregation,
        control_policy=goal.control_policy,
        control_policy_version=parent.control_policy_version,
        entity_match_version=parent.entity_match_version,
        locale=parent.locale,
        goals=[],
    )
