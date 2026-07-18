"""Reusable test doubles for prompt-driven role agents."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One complete structured-model invocation captured by the scripted fake."""

    purpose: str
    system_prompt: str
    user_payload: dict[str, Any] | BaseModel
    response_model: type[BaseModel]


class ScriptedModelClient:
    """Purpose-keyed FIFO model client used by unit and graph tests.

    Script values may be already-validated Pydantic objects, plain dictionaries, or
    exceptions.  Plain dictionaries are deliberately returned unchanged so the
    production agent remains responsible for validating its own output contract.
    """

    provider = "test-openai"
    model_name = "scripted"

    def __init__(
        self,
        responses: Mapping[str, Iterable[BaseModel | dict[str, Any] | Exception]]
        | None = None,
        *,
        provider: str = "test-openai",
        model_name: str = "scripted",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self._responses: defaultdict[
            str, deque[BaseModel | dict[str, Any] | Exception]
        ] = defaultdict(deque)
        self.calls: list[ModelCall] = []
        for purpose, values in (responses or {}).items():
            self._responses[purpose].extend(values)

    def queue(
        self,
        purpose: str,
        *responses: BaseModel | dict[str, Any] | Exception,
    ) -> None:
        """Append responses to one purpose's FIFO script."""

        self._responses[purpose].extend(responses)

    def calls_for(self, purpose: str) -> list[ModelCall]:
        return [call for call in self.calls if call.purpose == purpose]

    def count(self, purpose: str | None = None) -> int:
        return len(self.calls) if purpose is None else len(self.calls_for(purpose))

    def remaining(self, purpose: str) -> int:
        return len(self._responses[purpose])

    async def structured(
        self,
        system_prompt: str,
        user_payload: Mapping[str, Any] | BaseModel,
        response_model: type[BaseModel],
        purpose: str,
    ) -> BaseModel | dict[str, Any]:
        payload_copy: dict[str, Any] | BaseModel
        if isinstance(user_payload, BaseModel):
            payload_copy = user_payload.model_copy(deep=True)
        else:
            payload_copy = deepcopy(dict(user_payload))
        self.calls.append(
            ModelCall(
                purpose=purpose,
                system_prompt=system_prompt,
                user_payload=payload_copy,
                response_model=response_model,
            )
        )

        if not self._responses[purpose]:
            raise AssertionError(f"no scripted response remains for purpose {purpose!r}")
        response = self._responses[purpose].popleft()
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)
