"""TradePilot backend — FastAPI entrypoint.

Chạy dev: uv run uvicorn main:app --reload
Kiểm tra: http://localhost:8000/health  và  http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import auth, predictions, stocks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO Phase 1.4: khởi động scheduler (create_scheduler().start()) ở đây.
    yield


app = FastAPI(title="TradePilot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(stocks.router)
app.include_router(auth.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tradepilot-backend"}
