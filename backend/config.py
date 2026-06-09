"""Cấu hình app — đọc từ biến môi trường / .env (không hardcode secret)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    redis_url: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def sync_database_url(self) -> str:
        """URL sync (psycopg2) cho Alembic — derive từ database_url async (asyncpg)."""
        url = self.database_url
        if url.startswith("postgresql+asyncpg://"):
            return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if url.startswith("sqlite+aiosqlite://"):  # dùng cho autogen migration trên sqlite
            return url.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return url


settings = Settings()
