from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.llm as llm_module
from app.llm import (
    ModelInvocationError,
    ModelOutputContractError,
    NativeToolCall,
    OpenAIModelClient,
)
from app.schemas import CacheScope, Intent, PlannerDecision


PERSONS_TOOL = {
    "type": "function",
    "function": {
        "name": "persons",
        "description": "Search raw person records.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
FINISH_TOOL = {
    "type": "function",
    "name": "finish",
    "description": "Finish with verified records.",
    "parameters": {
        "type": "object",
        "properties": {"record_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["record_ids"],
        "additionalProperties": False,
    },
    "strict": True,
}


class _Runnable:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.invocations: list[tuple[object, object]] = []

    async def ainvoke(self, messages, config):
        self.invocations.append((messages, config))
        if self.error is not None:
            raise self.error
        return self.result


class _ChatModel:
    def __init__(self, *, result=None, error: Exception | None = None, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.runnable = _Runnable(result=result, error=error)
        self.bind_calls: list[tuple[object, object, object, object]] = []
        self.structured_calls: list[tuple[object, object, object]] = []

    def bind_tools(
        self,
        tools,
        *,
        tool_choice,
        strict,
        parallel_tool_calls,
    ):
        self.bind_calls.append(
            (tools, tool_choice, strict, parallel_tool_calls)
        )
        return self.runnable

    def with_structured_output(self, schema, *, method, include_raw, strict):
        self.structured_calls.append((schema, method, include_raw, strict))
        return self.runnable


def _client(monkeypatch, result=None, error: Exception | None = None):
    chat = _ChatModel(result=result, error=error)
    monkeypatch.setattr(llm_module, "ChatOpenAI", lambda **kwargs: chat)
    return (
        OpenAIModelClient(api_key="server-test-key", model_name="model-test"),
        chat,
    )


@pytest.mark.asyncio
async def test_researcher_uses_one_required_strict_native_tool_call(monkeypatch) -> None:
    response = SimpleNamespace(
        tool_calls=[
            {
                "name": "persons",
                "args": {"query": "Mask", "match_mode": "fuzzy"},
                "id": "call_123",
                "type": "tool_call",
            }
        ]
    )
    client, chat = _client(monkeypatch, response)

    result = await client.researcher_tool_call(
        "complete researcher prompt",
        {"current_query": "untrusted text"},
        [PERSONS_TOOL, FINISH_TOOL],
        "researcher",
    )

    assert result == NativeToolCall(
        name="persons",
        arguments={"query": "Mask", "match_mode": "fuzzy"},
        call_id="call_123",
    )
    wrapped_finish = {
        "type": "function",
        "function": {
            key: value for key, value in FINISH_TOOL.items() if key != "type"
        },
    }
    assert chat.bind_calls == [
        ([PERSONS_TOOL, wrapped_finish], "required", True, False)
    ]
    messages, config = chat.runnable.invocations[0]
    assert "untrusted text" in messages[1].content
    assert config["metadata"] == {
        "purpose": "researcher",
        "provider": "openai",
        "model_name": "model-test",
    }
    assert chat.structured_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_calls",
    [
        [],
        [
            {"name": "persons", "args": {"query": "a"}, "id": "one"},
            {"name": "finish", "args": {"record_ids": []}, "id": "two"},
        ],
        [{"name": "web_search", "args": {"query": "a"}, "id": "unknown"}],
        [{"name": "persons", "args": '{"query":"a"}', "id": "string-args"}],
    ],
    ids=["no-call", "multiple-calls", "unknown-tool", "non-object-arguments"],
)
async def test_researcher_rejects_invalid_native_tool_responses(
    monkeypatch,
    tool_calls,
) -> None:
    client, _ = _client(monkeypatch, SimpleNamespace(tool_calls=tool_calls))

    with pytest.raises(ModelOutputContractError) as caught:
        await client.researcher_tool_call(
            "complete researcher prompt",
            {"verified_records": []},
            [PERSONS_TOOL, FINISH_TOOL],
            "researcher",
        )

    assert caught.value.purpose == "researcher"
    assert caught.value.__cause__ is None
    assert "web_search" not in str(caught.value)
    assert "string-args" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_call",
    [
        "not-an-object",
        {"name": 42, "args": {}, "id": "bad-name"},
        {"name": "persons", "args": {}, "id": 42},
        {"name": "persons", "args": {}, "id": ""},
    ],
    ids=[
        "non-object-call",
        "invalid-name-type",
        "invalid-call-id-type",
        "local-pydantic-validation",
    ],
)
async def test_researcher_classifies_local_native_call_validation_as_contract_error(
    monkeypatch,
    tool_call,
) -> None:
    client, _ = _client(
        monkeypatch,
        SimpleNamespace(tool_calls=[tool_call]),
    )

    with pytest.raises(ModelOutputContractError) as caught:
        await client.researcher_tool_call(
            "complete researcher prompt",
            {"verified_records": []},
            [PERSONS_TOOL],
            "researcher",
        )

    assert caught.value.purpose == "researcher"
    assert caught.value.__cause__ is None
    assert "bad-name" not in str(caught.value)


@pytest.mark.asyncio
async def test_researcher_sanitizes_native_tool_provider_error(monkeypatch) -> None:
    provider_secret = "authorization failed for sk-sensitive-value"
    client, _ = _client(monkeypatch, error=RuntimeError(provider_secret))

    with pytest.raises(ModelInvocationError) as caught:
        await client.researcher_tool_call(
            "complete researcher prompt",
            {"verified_records": []},
            [PERSONS_TOOL],
            "researcher",
        )

    assert provider_secret not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_structured_is_json_schema_only_and_rejects_researcher(monkeypatch) -> None:
    planner_result = PlannerDecision.model_validate(
        {
            "intent": "get_company_profile",
            "entity_references": [
                {
                    "mention": "Example Labs",
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
                    "goal": "Resolve the fictional company.",
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
    client, chat = _client(monkeypatch, planner_result)

    result = await client.structured(
        "complete planner prompt",
        {"current_query": "Example Labs"},
        PlannerDecision,
        "planner",
    )

    assert result == planner_result
    assert chat.structured_calls == [
        (PlannerDecision, "json_schema", True, True)
    ]

    with pytest.raises(ValueError, match="researcher_tool_call"):
        await client.structured(
            "complete researcher prompt",
            {"current_query": "Example Labs"},
            PlannerDecision,
            "researcher",
        )


@pytest.mark.asyncio
async def test_invalid_tool_definitions_fail_before_provider_invocation(monkeypatch) -> None:
    client, chat = _client(monkeypatch, SimpleNamespace(tool_calls=[]))

    with pytest.raises(ValueError, match="must not be empty"):
        await client.researcher_tool_call(
            "complete researcher prompt",
            {},
            [],
            "researcher",
        )
    with pytest.raises(ValueError, match="must be unique"):
        await client.researcher_tool_call(
            "complete researcher prompt",
            {},
            [PERSONS_TOOL, PERSONS_TOOL],
            "researcher",
        )

    assert chat.bind_calls == []
