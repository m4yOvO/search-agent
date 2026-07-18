from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.prompts import (
    PLANNER_FEW_SHOT_EXAMPLES,
    PLANNER_PROMPT_VERSION,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_FEW_SHOT_EXAMPLES,
    RESEARCHER_PROMPT_VERSION,
    RESEARCHER_SYSTEM_PROMPT,
    VISUALIZER_FEW_SHOT_EXAMPLES,
    VISUALIZER_PROMPT_VERSION,
    VISUALIZER_SYSTEM_PROMPT,
)
from app.llm import NativeToolCall
from app.schemas import (
    PlannerDecision,
    QuerySignature,
    ResearcherDecision,
    ToolName,
    VisualizerDecision,
)
from app.tools.contracts import CompaniesRequest, PersonsRequest, RelationsRequest


def test_planner_examples_are_schema_valid_task_dags() -> None:
    assert len(PLANNER_FEW_SHOT_EXAMPLES) == 4
    decisions = [
        PlannerDecision.model_validate(example["output"])
        for example in PLANNER_FEW_SHOT_EXAMPLES
    ]

    single, multiple, ownership, control = decisions
    assert len(single.entity_references) == 1
    assert [task.tool for task in single.research_tasks] == [
        ToolName.PERSONS,
        ToolName.RELATIONS,
    ]
    broad = single.research_tasks[-1]
    assert broad.relation_types == []
    assert broad.raw_relation_types == []
    assert multiple.result_merge.value == "union"
    assert [task.tool for task in multiple.research_tasks[:2]] == [
        ToolName.COMPANIES,
        ToolName.PERSONS,
    ]
    assert all(task.depends_on for task in multiple.research_tasks[2:])
    ownership_relation = ownership.research_tasks[-1]
    assert [item.value for item in ownership_relation.relation_types] == ["owns"]
    assert ownership_relation.raw_relation_types == ["Owns"]
    assert ownership_relation.direction.value == "outgoing"
    assert control.intent.value == "find_controlled_companies"
    control_probe, control_fallback = control.research_tasks[-2:]
    assert [item.value for item in control_probe.relation_types] == ["controls"]
    assert set(item.value for item in control_fallback.relation_types) == {
        "founded",
        "works_at",
        "owns",
    }
    assert control_fallback.depends_on == [control_probe.task_id]


@pytest.mark.parametrize(
    ("index", "request_model"),
    [(0, PersonsRequest), (1, RelationsRequest)],
)
def test_researcher_examples_are_real_native_tool_calls(index, request_model) -> None:
    assert len(RESEARCHER_FEW_SHOT_EXAMPLES) == 2
    call = NativeToolCall.model_validate(RESEARCHER_FEW_SHOT_EXAMPLES[index]["output"])
    request_model.model_validate(call.arguments)


def test_visualizer_example_uses_minimal_verified_record_contract() -> None:
    assert len(VISUALIZER_FEW_SHOT_EXAMPLES) == 2
    decisions = [
        VisualizerDecision.model_validate(example["output"])
        for example in VISUALIZER_FEW_SHOT_EXAMPLES
    ]
    assert all(decision.answer for decision in decisions)
    assert decisions[0].answer_record_ids == ["relation:raw:9001"]
    assert decisions[1].answer_record_ids == [
        "relation:raw:9001",
        "relation:raw:9002",
    ]


def test_agent_decisions_reject_removed_legacy_projection_fields() -> None:
    with pytest.raises(ValidationError):
        ResearcherDecision.model_validate(
            {
                "action": "finish",
                "resolved_entities": {"虚构名称": "company:C900"},
            }
        )
    with pytest.raises(ValidationError):
        VisualizerDecision.model_validate(
            {
                "answer": "虚构回答。",
                "answer_record_ids": [],
                "node_ids": ["company:C900"],
            }
        )


def test_query_signature_does_not_silently_migrate_old_relation_fields() -> None:
    with pytest.raises(ValidationError):
        QuerySignature.model_validate(
            {
                "intent": "find_related_companies",
                "relation_types": ["founded"],
            }
        )
    with pytest.raises(ValidationError):
        QuerySignature.model_validate(
            {
                "intent": "find_controlled_companies",
                "relation_types": ["controls"],
                "requested_relation_types": ["controls"],
                "effective_relation_types": ["controls"],
            }
        )


def test_role_prompts_are_versioned_complete_and_fixture_answer_free() -> None:
    prompts = (
        (PLANNER_PROMPT_VERSION, PLANNER_SYSTEM_PROMPT),
        (RESEARCHER_PROMPT_VERSION, RESEARCHER_SYSTEM_PROMPT),
        (VISUALIZER_PROMPT_VERSION, VISUALIZER_SYSTEM_PROMPT),
    )
    for version, prompt in prompts:
        assert version.startswith("enterprise-agents-v23-task-dag:")
        assert f"提示词版本：{version}" in prompt
        for section in ("# 角色目标", "# 输入契约", "# 事实边界", "# 失败策略", "# 输出契约"):
            assert section in prompt
        for fixture_answer in ("Elon Musk", "Tesla, Inc.", "SpaceX", "xAI", "马云", "阿里巴巴集团"):
            assert fixture_answer not in prompt

    assert "entity_catalog" in PLANNER_SYSTEM_PROMPT
    assert "research_tasks" in PLANNER_SYSTEM_PROMPT
    assert "relation_types 与 raw_relation_types 留空" in PLANNER_SYSTEM_PROMPT
    assert "ready_task_contracts" in RESEARCHER_SYSTEM_PROMPT
    assert "为空表示查询全部直接关系" in RESEARCHER_SYSTEM_PROMPT
    assert "固定演示披露" in VISUALIZER_SYSTEM_PROMPT


def test_planner_provider_schema_is_small_and_has_no_legacy_phase_fields() -> None:
    fields = set(PlannerDecision.model_json_schema()["properties"])
    assert fields == {
        "intent",
        "entity_references",
        "research_tasks",
        "result_merge",
        "clarification_question",
        "query_requires_realtime_data",
    }
    assert {
        "association_operator",
        "control_policy",
        "suggested_tools",
        "typed_research_goals",
        "context_entity_ids",
    }.isdisjoint(fields)


def test_planner_rejects_unknown_task_dependencies_and_cycles() -> None:
    output = PLANNER_FEW_SHOT_EXAMPLES[0]["output"]
    unknown = {
        **output,
        "research_tasks": [
            {**output["research_tasks"][0], "depends_on": ["missing"]},
            output["research_tasks"][1],
        ],
    }
    with pytest.raises(ValidationError, match="unknown task"):
        PlannerDecision.model_validate(unknown)

    cyclic_tasks = [dict(item) for item in output["research_tasks"]]
    cyclic_tasks[0] = {**cyclic_tasks[0], "depends_on": ["t2"]}
    with pytest.raises(ValidationError, match="acyclic"):
        PlannerDecision.model_validate({**output, "research_tasks": cyclic_tasks})


def test_entity_request_schemas_remain_closed() -> None:
    with pytest.raises(ValidationError):
        PersonsRequest.model_validate({"query": "fictional", "unknown": True})
    with pytest.raises(ValidationError):
        CompaniesRequest.model_validate({"query": "fictional", "unknown": True})
