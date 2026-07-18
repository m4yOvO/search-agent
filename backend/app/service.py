"""Application service and resource lifecycle for the public API."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from .config import Settings
from .graph_store import GraphSnapshotStore
from .schemas import (
    CacheMetadata,
    ChatErrorCode,
    ChatRequest,
    ChatResponse,
    ChatStatus,
    GraphPayload,
    ReadyResponse,
    TraceMetadata,
)
from .state_views import request_semantics


logger = logging.getLogger(__name__)


class CompiledGraph(Protocol):
    async def ainvoke(
        self, input: Mapping[str, Any], config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class InvalidConversationId(ValueError):
    """Raised when a caller supplies a non-UUID conversation selector."""


class GraphExecutionError(RuntimeError):
    """Raised when the state graph fails or emits an invalid public result."""


class GraphExecutionTimeout(GraphExecutionError):
    """Raised when a complete multi-agent turn exceeds its bounded budget."""


@dataclass(slots=True)
class ServiceResources:
    graph: CompiledGraph
    graph_store: GraphSnapshotStore
    settings: Settings
    cache: Any | None = None
    tools: Any | None = None
    model: Any | None = None
    checkpointer: Any | None = None
    fixtures_ready: bool = True
    checkpoint_ready: bool = True


class ApplicationService:
    """Thin API-facing facade around one compiled LangGraph invocation."""

    def __init__(self, resources: ServiceResources) -> None:
        self.resources = resources
        self._conversation_locks: defaultdict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    @staticmethod
    def normalize_conversation_id(value: str | None) -> str:
        if value is None:
            return str(uuid4())
        try:
            return str(UUID(value))
        except (ValueError, AttributeError) as exc:
            raise InvalidConversationId("conversation_id must be a valid UUID") from exc

    async def chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id = self.normalize_conversation_id(request.conversation_id)
        request_id = f"request:{uuid4()}"
        graph_input = {
            "conversation_id": conversation_id,
            "request_id": request_id,
            "current_query": request.message,
            "locale": request.locale,
        }
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": self.resources.settings.graph_recursion_limit,
        }

        # State updates for the same thread must be sequential.  This is deliberately
        # process-local because the MVP explicitly runs one worker/replica.
        async with self._conversation_locks[conversation_id]:
            try:
                # The route owns no agent sequencing: one compiled StateGraph call is
                # the sole orchestration boundary.
                async with asyncio.timeout(
                    self.resources.settings.chat_timeout_seconds
                ):
                    result = await self.resources.graph.ainvoke(graph_input, config=config)
                response = self._build_response(
                    result, conversation_id=conversation_id, request_id=request_id
                )
                await self.resources.graph_store.save(conversation_id, response.graph)
            except TimeoutError as exc:
                logger.warning(
                    "state_graph_timeout",
                    extra={"event": "state_graph_timeout", "request_id": request_id},
                )
                raise GraphExecutionTimeout(
                    "state graph execution exceeded the configured turn timeout"
                ) from exc
            except GraphExecutionError:
                raise
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                logger.exception(
                    "state_graph_invalid_result",
                    extra={"event": "state_graph_invalid_result", "request_id": request_id},
                )
                raise GraphExecutionError(
                    "state graph returned an invalid public result"
                ) from exc
            except Exception as exc:
                logger.exception(
                    "state_graph_failed",
                    extra={"event": "state_graph_failed", "request_id": request_id},
                )
                raise GraphExecutionError("state graph execution failed") from exc

        if response.memory.cache_hit:
            logger.info(
                "cache_hit",
                extra={
                    "event": "cache_hit",
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "match_type": response.memory.match_type,
                    "cache_status": response.memory.status,
                    "result_id": response.memory.result_id,
                },
            )
        else:
            logger.info(
                "cache_miss",
                extra={
                    "event": "cache_miss",
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "write_operation": response.memory.write_operation,
                },
            )
        logger.info(
            "chat_completed",
            extra={
                "event": "chat_completed",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "model_provider": response.trace.model_provider,
                "model_name": response.trace.model_name,
                "model_calls": response.trace.model_calls,
                "planner_model_calls": response.trace.planner_model_calls,
                "researcher_model_calls": response.trace.researcher_model_calls,
                "visualizer_model_calls": response.trace.visualizer_model_calls,
                "tool_calls": response.trace.tool_calls,
                "research_steps": response.trace.research_steps,
                "replans": response.trace.replans,
                "no_match": bool(result.get("no_match")),
                # Keep completion logs aggregate-only.  The public response exposes
                # validated typed steps, but logging the full sequence would add no
                # operational value and would widen the accidental disclosure surface.
                "route_count": len(response.trace.route_history),
                "agent_step_count": len(response.trace.agent_steps),
            },
        )
        return response

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        raise TypeError("state graph output must be a mapping")

    def _build_response(
        self,
        raw_result: Any,
        *,
        conversation_id: str,
        request_id: str,
    ) -> ChatResponse:
        result = self._as_mapping(raw_result)
        graph_value = result.get("session_graph")
        if graph_value is None:
            raise KeyError("session_graph")
        graph = GraphPayload.model_validate(graph_value)
        declared_graph_id = result.get("graph_id")
        if declared_graph_id is not None and declared_graph_id != graph.graph_id:
            raise ValueError("graph_id does not match session_graph.graph_id")

        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")

        memory_value = result.get("cache_metadata", result.get("memory", {}))
        trace_value = result.get("trace")
        if trace_value is None:
            trace_value = {
                "researcher_invoked": result.get("researcher_invoked", False),
                "tool_calls": result.get("tool_call_count", 0),
                "research_steps": result.get(
                    "research_step_count", result.get("research_steps", 0)
                ),
                "replans": result.get("replan_count", 0),
                "model_provider": result.get("model_provider", "openai"),
                "model_name": result.get("model_name"),
                "model_calls": result.get("model_call_count", 0),
                "planner_model_calls": result.get("planner_model_calls", 0),
                "researcher_model_calls": result.get("researcher_model_calls", 0),
                "visualizer_model_calls": result.get("visualizer_model_calls", 0),
                "prompt_versions": _prompt_versions(),
                "route_history": result.get("route_history", []),
                "agent_steps": result.get("agent_steps", []),
            }
        else:
            trace_value = dict(self._as_mapping(trace_value))
            # ``agent_steps`` is a request-transient StateGraph channel.  Prefer that
            # channel whenever it is present so an older upstream aggregate ``trace``
            # object cannot silently omit the newly required public audit steps.
            if "agent_steps" in result:
                trace_value["agent_steps"] = result["agent_steps"]

        return ChatResponse(
            conversation_id=conversation_id,
            request_id=request_id,
            status=self._chat_status(result),
            error_code=self._safe_error_code(result),
            answer=answer.strip(),
            graph_id=graph.graph_id,
            graph=graph,
            memory=CacheMetadata.model_validate(memory_value or {}),
            trace=TraceMetadata.model_validate(trace_value),
        )

    @staticmethod
    def _chat_status(result: Mapping[str, Any]) -> ChatStatus:
        if result.get("run_status") == "failed":
            return ChatStatus.FAILED
        if request_semantics(result).needs_clarification:
            return ChatStatus.CLARIFICATION
        return ChatStatus.SUCCESS

    @staticmethod
    def _safe_error_code(result: Mapping[str, Any]) -> ChatErrorCode | None:
        if result.get("run_status") != "failed":
            return None
        if result.get("planner_failed"):
            return ChatErrorCode.PLANNING_FAILURE
        if result.get("llm_errors"):
            return ChatErrorCode.MODEL_FAILURE
        if result.get("research_failure_reason"):
            return ChatErrorCode.RESEARCH_FAILURE
        if result.get("tool_errors"):
            return ChatErrorCode.TOOL_FAILURE
        return ChatErrorCode.AGENT_FAILURE

    async def get_graph(
        self, *, graph_id: str | None = None, conversation_id: str | None = None
    ) -> GraphPayload | None:
        if (graph_id is None) == (conversation_id is None):
            raise ValueError("provide exactly one of graph_id or conversation_id")
        if conversation_id is not None:
            normalized = self.normalize_conversation_id(conversation_id)
            return await self.resources.graph_store.get_latest_for_conversation(normalized)
        assert graph_id is not None
        return await self.resources.graph_store.get_by_graph_id(graph_id)

    async def ready(self) -> ReadyResponse:
        checks = {
            "fixtures": bool(self.resources.fixtures_ready),
            "checkpoint": bool(self.resources.checkpoint_ready)
            and await _ping_checkpoint(self.resources.checkpointer),
            "state_graph": callable(getattr(self.resources.graph, "ainvoke", None)),
            "graph_store": await self.resources.graph_store.ping(),
            "chroma": await _ping_component(self.resources.cache),
            "openai_config": bool(
                self.resources.model is not None
                and getattr(self.resources.model, "model_name", "")
            ),
        }
        return ReadyResponse(
            status="ready" if all(checks.values()) else "not_ready", checks=checks
        )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _ping_component(component: Any | None) -> bool:
    if component is None:
        return False
    ping = getattr(component, "ping", None)
    if not callable(ping):
        return False
    try:
        return bool(await _maybe_await(ping()))
    except Exception:
        return False


async def _ping_checkpoint(checkpointer: Any | None) -> bool:
    if checkpointer is None:
        return False
    connection = getattr(checkpointer, "conn", None)
    execute = getattr(connection, "execute", None)
    if not callable(execute):
        # A successful setup in the lifespan is still meaningful for saver
        # implementations that intentionally hide their connection.
        return True
    try:
        cursor = await _maybe_await(execute("SELECT 1"))
        fetchone = getattr(cursor, "fetchone", None)
        row = await _maybe_await(fetchone()) if callable(fetchone) else (1,)
        close = getattr(cursor, "close", None)
        if callable(close):
            await _maybe_await(close())
        return row is not None
    except Exception:
        return False


async def _close_component(component: Any | None) -> None:
    close = getattr(component, "close", None)
    if callable(close):
        await _maybe_await(close())


def _prompt_versions() -> dict[str, str]:
    from .agents.graph import PROMPT_VERSIONS

    return dict(PROMPT_VERSIONS)


@asynccontextmanager
async def open_application_service(
    settings: Settings, *, model_client: Any | None = None
) -> AsyncIterator[ApplicationService]:
    """Create all production resources with a single, explicit owner.

    Imports are intentionally local: API unit tests can inject a compiled graph without
    initializing Chroma or importing the implementation modules.
    """

    from .agents.graph import AgentDependencies, compile_agent_graph
    from .llm import OpenAIModelClient
    from .memory.checkpoint import open_checkpointer
    from .memory import LongTermMemory
    from .tools import FixtureRepository, ToolRegistry

    settings.ensure_runtime_directories()
    graph_store = await GraphSnapshotStore.open(settings.graph_store_path)
    cache: Any | None = None
    tools: Any | None = None
    try:
        if model_client is None:
            api_key = settings.require_openai_api_key()
            model = OpenAIModelClient(
                api_key=api_key,
                model_name=settings.openai_model,
                timeout_seconds=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        else:
            # Explicit dependency injection is used by tests only. There is no
            # environment-controlled runtime fallback from OpenAI to a scripted model.
            model = model_client
        repository = FixtureRepository.load(settings.data_directory)
        repository.assert_ready()
        tools = ToolRegistry(repository)
        planner_catalog = repository.compact_planner_catalog()
        planner_tools = tuple(
            {"name": spec.name.value, "description": spec.description}
            for spec in tools.specs
        )
        cache = LongTermMemory(settings, data_version=repository.data_version)
        # Long-term memory is a fail-open query cache.  Its implementation records a
        # connection failure and lets the graph continue through Researcher.
        await _maybe_await(cache.initialize())
        async with open_checkpointer(settings.checkpoint_path) as checkpointer:
            dependencies = AgentDependencies(
                settings=settings,
                tools=tools,
                cache=cache,
                data_version=repository.data_version,
                model=model,
                session_graph_validator=repository.session_graph_is_trusted,
                planner_catalog=planner_catalog,
                planner_tools=planner_tools,
            )
            compiled = compile_agent_graph(dependencies, checkpointer=checkpointer)
            resources = ServiceResources(
                graph=compiled,
                graph_store=graph_store,
                settings=settings,
                cache=cache,
                tools=tools,
                model=model,
                checkpointer=checkpointer,
                fixtures_ready=bool(repository.is_loaded),
                checkpoint_ready=True,
            )
            yield ApplicationService(resources)
    finally:
        await _close_component(cache)
        await _close_component(tools)
        await graph_store.close()
