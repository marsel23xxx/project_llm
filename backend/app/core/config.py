from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Agentic Data Analyst"
    environment: Literal["development", "production"] = "development"
    frontend_origin: str = "http://localhost:3000"

    database_url: str = "postgresql+asyncpg://analyst:analyst@localhost:5432/analyst"
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    login_max_attempts: int = 5
    login_window_seconds: int = 900
    registration_open: bool = True
    register_max_per_hour: int = 5

    # Data upload
    data_dir: str = "/data"
    max_upload_mb: int = 50
    dataset_ttl_hours: int = 24

    # LLM
    # local  : semua jawaban dari LLM lokal (gratis, untuk development)
    # hybrid : pertanyaan sederhana ke lokal, multi-langkah ke Claude API
    # api    : semua ke Claude API, lokal sebagai cadangan
    llm_mode: Literal["local", "hybrid", "api"] = "local"
    local_llm_url: str = "http://localhost:8080"
    local_llm_model: str = "qwen2.5-coder-3b-instruct"
    # Hosted OpenAI-compatible API (DeepSeek by default). Used when ANTHROPIC_API_KEY is empty.
    remote_llm_base_url: str = "https://api.deepseek.com"
    remote_llm_api_key: str | None = None
    remote_llm_model: str = "deepseek-flash"
    remote_llm_disable_thinking: bool = True
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5-5"
    # Kosongkan (CLAUDE_EFFORT=) untuk model yang tidak mendukung effort, mis. Haiku 4.5
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    daily_question_quota: int = 20

    # Agent
    sandbox_url: str = "http://localhost:8001"
    forecast_url: str = "http://localhost:8002"
    sql_timeout_seconds: float = 15.0
    max_result_rows: int = 1000
    agent_max_attempts: int = 3
    history_messages: int = 6
    # With a hosted LLM, documents up to this many words are sent whole with each question
    # (about 35k tokens; larger ones fall back to retrieving passages).
    doc_full_context_words: int = 20000

    # ML models (backend/models)
    models_dir: str = "models"
    # Empty = use the threshold stored with the trained model.
    intent_min_confidence: float | None = None

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @field_validator("claude_effort", "anthropic_api_key", "remote_llm_api_key", "intent_min_confidence", mode="before")
    @classmethod
    def _empty_as_none(cls, value):
        return value or None

    @model_validator(mode="after")
    def _check_secrets(self) -> "Settings":
        if self.is_production and self.jwt_secret == "change-me":
            raise ValueError("JWT_SECRET wajib diganti di production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
