from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic,ollama"  # comma list = fallback chain
    anthropic_api_key: str = ""
    anthropic_chat_model: str = "claude-opus-5-5"
    anthropic_extract_model: str = "claude-haiku-4-5"
    anthropic_chat_effort: str = "low"  # Opus 5.5 always thinks; low effort keeps chat fast. "" disables.
    anthropic_api_url: str = "https://api.anthropic.com"  # explicit: ignore a stray ANTHROPIC_BASE_URL
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    llm_timeout: float = 60.0

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "brain-pass"
    sqlite_path: str = "brain.db"

    classifier: str = "rules"  # rules | laya
    memory_update_mode: str = "background"  # background | inline

    recent_turns: int = 6
    max_memories: int = 5
    context_char_budget: int = 1500
    memory_half_life_days: float = 180.0
    min_memory_confidence: float = 0.6
    max_message_chars: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()
