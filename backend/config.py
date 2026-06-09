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


settings = Settings()
