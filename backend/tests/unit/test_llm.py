from __future__ import annotations

import pytest

import app.llm as llm_module
from app.llm import ModelInvocationError, ModelOutputContractError, OpenAIModelClient
from app.schemas import CacheScope, Intent, PlannerDecision, ResearcherDecision


class _Runnable:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.invocations = []

    async def ainvoke(self, messages, config):
        self.invocations.append((messages, config))
        if self.error:
            raise self.error
        return self.result


class _ChatModel:
    def __init__(self, *, result=None, error: Exception | None = None, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.runnable = _Runnable(result=result, error=error)
        self.structured_calls = []

    def with_structured_output(self, schema, *, method, include_raw, strict):
        self.structured_calls.append((schema, method, include_raw, strict))
        return self.runnable


@pytest.mark.asyncio
async def test_openai_adapter_uses_role_appropriate_structured_methods(monkeypatch) -> None:
    planner_result = PlannerDecision.model_validate(
        {
            "intent": "get_company_profile",
            "entity_references": [
                {
                    "mention": "example",
                    "canonical_name": "Example Labs",
                    "source": "current_query",
                    "role": "subject",
                    "expected_types": ["company"],
                    "context_entity_id": None,
                }
            ],
            "research_tasks": [
                {
                    "task_id": "profile",
                    "goal": "Resolve a fictional company profile.",
                    "tool": "companies",
                    "subject_reference_indexes": [0],
                    "object_reference_indexes": [],
                    "relation_types": [],
                    "raw_relation_types": [],
                    "direction": "not_applicable",
                    "target_types": ["company"],
                    "requested_attributes": [],
                    "depends_on": [],
                }
            ],
            "result_merge": "not_applicable",
            "clarification_question": None,
            "query_requires_realtime_data": False,
        }
    )
    chat = _ChatModel(result=planner_result)
    monkeypatch.setattr(llm_module, "ChatOpenAI", lambda **kwargs: chat)
    client = OpenAIModelClient(api_key="server-test-key", model_name="model-test")

    result = await client.structured(
        "complete planner prompt",
        {"current_query": "untrusted user data"},
        PlannerDecision,
        "planner",
    )

    assert result == planner_result
    assert chat.structured_calls == [
        (PlannerDecision, "json_schema", True, True)
    ]
    assert chat.runnable.invocations[0][1]["metadata"]["purpose"] == "planner"

    researcher_chat = _ChatModel(result=None)
    monkeypatch.setattr(llm_module, "ChatOpenAI", lambda **kwargs: researcher_chat)
    researcher_client = OpenAIModelClient(
        api_key="server-test-key", model_name="model-test"
    )
    with pytest.raises(ValueError, match="researcher_tool_call"):
        await researcher_client.structured(
            "complete researcher prompt",
            {"verified_mock_records": []},
            ResearcherDecision,
            "researcher",
        )
    assert researcher_chat.structured_calls == []


@pytest.mark.asyncio
async def test_openai_adapter_sanitizes_provider_errors(monkeypatch) -> None:
    provider_detail = "authorization failed for sk-sensitive-value"
    chat = _ChatModel(error=RuntimeError(provider_detail))
    monkeypatch.setattr(llm_module, "ChatOpenAI", lambda **kwargs: chat)
    client = OpenAIModelClient(api_key="server-test-key", model_name="model-test")

    with pytest.raises(ModelInvocationError) as caught:
        await client.structured(
            "complete planner prompt",
            {"current_query": "hello"},
            PlannerDecision,
            "planner",
        )

    assert provider_detail not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_openai_adapter_classifies_safe_structured_parse_failures(
    monkeypatch,
) -> None:
    chat = _ChatModel(
        result={
            "raw": object(),
            "parsed": None,
            "parsing_error": ValueError("untrusted provider output"),
        }
    )
    monkeypatch.setattr(llm_module, "ChatOpenAI", lambda **kwargs: chat)
    client = OpenAIModelClient(api_key="server-test-key", model_name="model-test")

    with pytest.raises(ModelOutputContractError) as caught:
        await client.structured(
            "complete planner prompt",
            {"current_query": "hello"},
            PlannerDecision,
            "planner",
        )

    assert "untrusted provider output" not in str(caught.value)
    assert caught.value.__cause__ is None
