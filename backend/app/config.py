"""Runtime configuration with safe local defaults and environment overrides."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    agent_mode: str = "openai"
    log_level: str = "INFO"
    # Local development reads the three raw JSON files directly from data/.
    # Docker uses the same /app/data layout.
    data_directory: Path = PROJECT_ROOT / "data"
    runtime_directory: Path = PROJECT_ROOT / "runtime"

    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    chroma_collection_prefix: str = "enterprise_query_cache_v18"
    chroma_connect_retries: int = 12
    chroma_retry_delay_seconds: float = 1.0

    checkpoint_path: Path = PROJECT_ROOT / "runtime" / "checkpoints.sqlite3"
    graph_store_path: Path = PROJECT_ROOT / "runtime" / "graphs.sqlite3"

    short_term_max_turns: int = 15
    short_term_compact_oldest: int = 10
    short_term_keep_recent: int = 5

    cache_ttl_hours: int = 24
    graph_schema_version: int = 1
    # v3 adds association operators to requested/effective relation semantics,
    # raw qualifiers, verified
    # empty scopes, control policy, and entity-match algorithm provenance.
    query_signature_version: int = 4
    permission_scope: str = "public-demo"

    # ``max_research_steps`` is the normal Researcher model/tool-decision budget.
    # A bounded reserve is unlocked only after an accepted replan or one contract
    # correction, while ``agent_max_iterations`` remains the absolute hard ceiling.
    max_research_steps: int = Field(default=12, ge=1, le=32)
    max_tool_calls: int = Field(default=10, ge=1, le=32)
    max_replans: int = Field(default=2, ge=0, le=4)
    research_retry_step_allowance: int = Field(default=3, ge=1, le=8)
    agent_max_iterations: int = Field(default=20, ge=2, le=48)
    graph_recursion_limit: int = Field(default=64, ge=16, le=128)
    tool_timeout_seconds: float = 5.0
    chat_timeout_seconds: float = Field(default=570.0, gt=0, le=590)

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_timeout_seconds: float = 45.0
    openai_max_retries: int = 2

    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )

    @property
    def chroma_collection_name(self) -> str:
        return self.chroma_collection_prefix

    def ensure_runtime_directories(self) -> None:
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_store_path.parent.mkdir(parents=True, exist_ok=True)

    def require_openai_api_key(self) -> str:
        """Return the configured server-side key or fail before serving traffic."""

        if self.agent_mode.casefold() != "openai":
            raise ValueError("AGENT_MODE must be 'openai' for this MVP")
        if not self.openai_model.strip():
            raise ValueError("OPENAI_MODEL must not be empty")
        value = (
            self.openai_api_key.get_secret_value().strip()
            if self.openai_api_key is not None
            else ""
        )
        if not value:
            raise ValueError(
                "OPENAI_API_KEY is required; add it to the ignored project .env file"
            )
        return value


def get_settings() -> Settings:
    return Settings()
