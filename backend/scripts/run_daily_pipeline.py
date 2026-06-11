"""Entrypoint pipeline 16:00 — launchd gọi mỗi ngày làm việc lúc 16:05.

Chạy tay:
    cd backend && DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib" \\
        uv run --group inference python -m scripts.run_daily_pipeline

launchd: ~/Library/LaunchAgents/com.tradepilot.daily.plist (template: scripts/launchd/).
Exit code: 0 nếu mọi bước ok, 1 nếu có bước fail (launchd log được trạng thái).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from logging_config import configure_logging, init_sentry  # noqa: E402 — sau sys.path setup


def main() -> int:
    configure_logging()
    init_sentry()  # lỗi pipeline (logger.exception) tự lên Sentry khi có DSN

    from services.scheduler import run_daily_pipeline

    results = asyncio.run(run_daily_pipeline())
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
