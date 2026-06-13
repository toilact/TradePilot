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
    # Autolabel M8: key phụ để xoay vòng khi key chính hết quota/NGÀY (free tier ~20 req/ngày).
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    app_env: str = "development"
    log_level: str = "INFO"
    # Observability (M4) — rỗng = tắt, app vẫn chạy bình thường
    sentry_dsn: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # M5: APScheduler in-app (default TẮT — trigger chính là launchd 16:05 trên Mac)
    enable_scheduler: bool = False
    # M7: CORS production — danh sách origin cách nhau dấu phẩy; regex cho Vercel preview
    # (rỗng = tắt; Render set ^https://trade-pilot-kato-.*\.vercel\.app$ — không mở *.vercel.app)
    allowed_origins: str = "http://localhost:3000"
    allowed_origin_regex: str = ""

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def gemini_api_keys(self) -> list[str]:
        """Tất cả Gemini key (chính + phụ), khử rỗng/trùng, giữ thứ tự — để xoay vòng quota."""
        seen: set[str] = set()
        keys = [self.gemini_api_key, self.gemini_api_key_2, self.gemini_api_key_3]
        return [k for k in (k.strip() for k in keys) if k and not (k in seen or seen.add(k))]

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
