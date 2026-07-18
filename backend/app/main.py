"""FastAPI entry point for the enterprise relationship explorer."""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .logging_config import configure_logging
from .schemas import (
    ChatRequest,
    ChatResponse,
    GraphPayload,
    HealthResponse,
    ReadyResponse,
)
from .service import (
    ApplicationService,
    GraphExecutionError,
    GraphExecutionTimeout,
    InvalidConversationId,
    open_application_service,
)


logger = logging.getLogger(__name__)
ServiceFactory = Callable[[], AbstractAsyncContextManager[ApplicationService]]


def create_app(
    *,
    settings: Settings | None = None,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()

    if service_factory is None:

        def service_factory() -> AbstractAsyncContextManager[ApplicationService]:
            return open_application_service(runtime_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(runtime_settings.log_level)
        logger.info("application_starting", extra={"event": "application_starting"})
        assert service_factory is not None
        async with service_factory() as service:
            application.state.service = service
            yield
        logger.info("application_stopped", extra={"event": "application_stopped"})

    application = FastAPI(
        title="Enterprise Relationship Intelligence Explorer",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    def service_from(request: Request) -> ApplicationService:
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="service is not initialized")
        return service

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: {"model": ReadyResponse}},
    )
    async def ready(request: Request) -> ReadyResponse | JSONResponse:
        result = await service_from(request).ready()
        if result.status != "ready":
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        return result

    @application.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            return await service_from(request).chat(payload)
        except InvalidConversationId as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GraphExecutionTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except GraphExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @application.get("/graph", response_model=GraphPayload)
    async def graph(
        request: Request,
        graph_id: str | None = Query(default=None, min_length=1, max_length=128),
        conversation_id: str | None = Query(default=None, min_length=1, max_length=64),
    ) -> GraphPayload:
        if (graph_id is None) == (conversation_id is None):
            raise HTTPException(
                status_code=422,
                detail="provide exactly one of graph_id or conversation_id",
            )
        try:
            result = await service_from(request).get_graph(
                graph_id=graph_id, conversation_id=conversation_id
            )
        except InvalidConversationId as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="graph snapshot not found")
        return result

    return application


app = create_app()
