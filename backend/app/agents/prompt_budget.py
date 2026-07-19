"""Deterministic input budgeting for structured Planner prompts.

The budget covers the complete provider-facing contract: system instructions,
the compact runtime JSON payload, and the structured-output JSON Schema.  Only
bounded conversation prose is eligible for trimming; current-query and typed
contract data are never discarded here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel


__all__ = [
    "PromptBudgetExceeded",
    "PromptBudgetResult",
    "apply_prompt_budget",
]


try:
    _ENCODING = tiktoken.get_encoding("o200k_base")
except Exception:
    # A container image preloads this vocabulary.  The conservative fallback
    # keeps local/offline diagnostics available if that cache is unavailable;
    # it intentionally overestimates typical English and Chinese prompt text.
    _ENCODING = None


class PromptBudgetExceeded(ValueError):
    """Raised when protected prompt content alone exceeds the configured budget."""

    def __init__(self, *, estimated_tokens: int, max_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens
        super().__init__(
            "prompt exceeds its token budget after all eligible history was trimmed "
            f"({estimated_tokens} > {max_tokens})"
        )


@dataclass(frozen=True, slots=True)
class PromptBudgetResult:
    """A copied, budget-safe payload and its deterministic trimming counters."""

    payload: dict[str, Any]
    estimated_tokens: int
    trimmed_assistant_fields: int
    trimmed_user_fields: int


def apply_prompt_budget(
    system_prompt: str,
    payload: Mapping[str, Any],
    response_model: type[BaseModel],
    max_tokens: int,
) -> PromptBudgetResult:
    """Return a copied payload that fits, trimming only oldest turn prose.

    Assistant fields are cleared from oldest to newest before any user field is
    considered.  The estimate is recomputed after every individual field change,
    so no additional conversation text is removed once the budget is satisfied.
    """

    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if not isinstance(response_model, type) or not issubclass(
        response_model, BaseModel
    ):
        raise TypeError("response_model must be a Pydantic model class")

    copied_payload = deepcopy(dict(payload))
    response_schema = response_model.model_json_schema()
    estimated = _estimate_tokens(system_prompt, copied_payload, response_schema)
    if estimated <= max_tokens:
        return PromptBudgetResult(copied_payload, estimated, 0, 0)

    turns = copied_payload.get("recent_visible_turns")
    eligible_turns = turns if isinstance(turns, list) else []
    trimmed_assistant = 0
    trimmed_user = 0

    for field_name in ("assistant", "user"):
        for index, turn in enumerate(eligible_turns):
            updated_turn = _clear_nonempty_field(turn, field_name)
            if updated_turn is None:
                continue
            eligible_turns[index] = updated_turn
            if field_name == "assistant":
                trimmed_assistant += 1
            else:
                trimmed_user += 1
            estimated = _estimate_tokens(
                system_prompt, copied_payload, response_schema
            )
            if estimated <= max_tokens:
                return PromptBudgetResult(
                    copied_payload,
                    estimated,
                    trimmed_assistant,
                    trimmed_user,
                )

    raise PromptBudgetExceeded(
        estimated_tokens=estimated,
        max_tokens=max_tokens,
    )


def _clear_nonempty_field(turn: Any, field_name: str) -> Any | None:
    """Return a copied turn with one non-empty text field cleared, if possible."""

    if isinstance(turn, MutableMapping):
        value = turn.get(field_name)
        if not isinstance(value, str) or not value:
            return None
        updated = deepcopy(turn)
        updated[field_name] = ""
        return updated
    if isinstance(turn, BaseModel):
        value = getattr(turn, field_name, None)
        if not isinstance(value, str) or not value:
            return None
        return turn.model_copy(deep=True, update={field_name: ""})
    return None


def _estimate_tokens(
    system_prompt: str,
    payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> int:
    components = (
        system_prompt,
        _compact_json(payload),
        _compact_json(response_schema),
    )
    return sum(_component_tokens(component) for component in components)


def _component_tokens(component: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(component))
    byte_count = len(component.encode("utf-8"))
    return max(1, (byte_count + 1) // 2)


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")
