# TradePilot API — Render free 512MB (M7).
# KHÔNG cài group inference (torch/lightgbm) — inference chạy trên Mac qua launchd (ADR 0002/0003);
# Render chỉ serve read API từ predictions đã ghi sẵn trong Supabase.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Layer deps tách riêng để cache khi chỉ đổi code
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Render bơm $PORT lúc runtime; local fallback 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
