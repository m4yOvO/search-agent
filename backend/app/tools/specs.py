"""Reusable specifications for validated OpenAI-compatible local tools."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.schemas import Evidence, ToolName
from app.tools.contracts import ToolResultMeta, TypedToolResult


RequestT = TypeVar("RequestT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolHandlerOutput:
    records: tuple[dict[str, Any], ...]
    evidence: tuple[Evidence, ...]
    meta: ToolResultMeta


ToolHandler = Callable[[RequestT], Awaitable[ToolHandlerOutput]]
ToolResultAdapter = Callable[[ToolName, ToolHandlerOutput, int], TypedToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec(Generic[RequestT]):
    """The four parts of one tool: description, args, handler, adapter."""

    name: ToolName
    description: str
    request_model: type[RequestT]
    handler: ToolHandler[RequestT]
    result_adapter: ToolResultAdapter
    hidden_schema_fields: frozenset[str] = frozenset()

    async def invoke(
        self,
        arguments: RequestT | dict[str, Any],
        *,
        started_ns: int,
    ) -> TypedToolResult:
        request = (
            arguments
            if isinstance(arguments, self.request_model)
            else self.request_model.model_validate(arguments)
        )
        output = await self.handler(request)
        return self.result_adapter(self.name, output, started_ns)

    def openai_function_schema(self) -> dict[str, Any]:
        """Return an OpenAI Responses API strict function-tool definition."""

        parameters = copy.deepcopy(self.request_model.model_json_schema())
        properties = parameters.get("properties", {})
        for field_name in self.hidden_schema_fields:
            properties.pop(field_name, None)
        make_openai_strict_schema(parameters)
        return {
            "type": "function",
            "name": self.name.value,
            "description": self.description,
            "parameters": parameters,
            "strict": True,
        }

    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """Return the equivalent Chat Completions wrapper when needed."""

        function = self.openai_function_schema()
        return {
            "type": "function",
            "function": {key: value for key, value in function.items() if key != "type"},
        }


def create_tool_spec(
    *,
    name: ToolName,
    description: str,
    request_model: type[RequestT],
    handler: ToolHandler[RequestT],
    result_adapter: ToolResultAdapter,
    hidden_schema_fields: frozenset[str] = frozenset(),
) -> ToolSpec[RequestT]:
    """Factory kept deliberately small so every tool follows the same boundary."""

    if not description.strip():
        raise ValueError(f"tool {name.value} requires a non-empty description")
    return ToolSpec(
        name=name,
        description=description.strip(),
        request_model=request_model,
        handler=handler,
        result_adapter=result_adapter,
        hidden_schema_fields=hidden_schema_fields,
    )


def make_openai_strict_schema(schema: dict[str, Any]) -> None:
    """Convert a Pydantic schema to OpenAI's strict JSON-schema subset in place.

    Pydantic may emit a local ``$ref`` with sibling annotations such as
    ``description``. OpenAI rejects that valid general JSON Schema construct for
    strict function tools, so local definitions are expanded before closing every
    object. The request model remains the execution-time source of truth.
    """

    inlined = _inline_local_refs(schema, root=schema)
    schema.clear()
    schema.update(inlined)
    _make_openai_strict(schema)


def _inline_local_refs(value: Any, *, root: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_inline_local_refs(item, root=root) for item in value]
    if not isinstance(value, dict):
        return value

    reference = value.get("$ref")
    if isinstance(reference, str):
        prefix = "#/$defs/"
        if not reference.startswith(prefix):
            raise ValueError("only local Pydantic $defs references are supported")
        definition_name = reference.removeprefix(prefix)
        definitions = root.get("$defs")
        if not isinstance(definitions, dict) or definition_name not in definitions:
            raise ValueError("Pydantic schema contains an unresolved local reference")
        target = _inline_local_refs(
            copy.deepcopy(definitions[definition_name]),
            root=root,
        )
        if not isinstance(target, dict):
            raise ValueError("Pydantic local schema definition must be an object")
        for key, sibling in value.items():
            if key != "$ref":
                target[key] = _inline_local_refs(sibling, root=root)
        return target

    return {
        key: _inline_local_refs(item, root=root)
        for key, item in value.items()
        if key != "$defs"
    }


def _make_openai_strict(schema: dict[str, Any]) -> None:
    """Apply recursive strict object requirements after local refs are expanded."""

    # Defaults are an application concern; OpenAI strict function schemas require
    # the model to emit every property explicitly and do not need default keywords.
    schema.pop("default", None)
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        schema["required"] = list(properties)
    for value in schema.values():
        if isinstance(value, dict):
            _make_openai_strict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _make_openai_strict(item)
