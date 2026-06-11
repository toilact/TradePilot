"""Test run_daily_pipeline (M5) — steps/notify inject, không network/model.

Bất biến kiểm tra:
- 1 bước fail KHÔNG giết pipeline: các bước sau vẫn chạy, notifier vẫn được gọi.
- Tổng kết phản ánh đúng trạng thái (❌ fail / ⚠️ warning / ✅ ok).
- notify lỗi cũng không làm pipeline raise.
- Exit semantics: results đủ cho caller tính exit code (all ok → 0).
"""

from __future__ import annotations

import pytest

from services.scheduler import StepResult, build_summary, run_daily_pipeline


def _ok_step(name: str, detail: str = "ok"):
    async def step() -> StepResult:
        return StepResult(name, True, detail)

    step.__name__ = f"_step_{name}"
    return step


def _fail_step(name: str):
    async def step() -> StepResult:
        raise RuntimeError("nổ giữa chừng")

    step.__name__ = f"_step_{name}"
    return step


class _Notify:
    def __init__(self, raise_exc: bool = False):
        self.messages: list[str] = []
        self.raise_exc = raise_exc

    async def __call__(self, text: str) -> bool:
        if self.raise_exc:
            raise RuntimeError("telegram sập")
        self.messages.append(text)
        return True


async def test_mot_buoc_fail_cac_buoc_sau_van_chay():
    ran: list[str] = []

    def tracking(name: str):
        async def step() -> StepResult:
            ran.append(name)
            return StepResult(name, True, "ok")

        step.__name__ = f"_step_{name}"
        return step

    notify = _Notify()
    results = await run_daily_pipeline(
        steps=(tracking("prices"), _fail_step("news"), tracking("inference")), notify=notify
    )

    assert ran == ["prices", "inference"]  # bước sau bước fail VẪN chạy
    assert [r.ok for r in results] == [True, False, True]
    assert "RuntimeError" in results[1].detail
    assert len(notify.messages) == 1  # notifier vẫn được gọi
    assert "❌" in notify.messages[0] and "news" in notify.messages[0]


async def test_systemexit_khong_giet_pipeline():
    """vnstock rate-limit thoát bằng SystemExit (không phải Exception) — bài học DoD M5:
    lần chạy thật đầu tiên chết giữa bước 1, không Telegram. Pipeline phải sống sót."""

    async def exit_step() -> StepResult:
        raise SystemExit(1)

    exit_step.__name__ = "_step_prices"
    notify = _Notify()
    results = await run_daily_pipeline(steps=(exit_step, _ok_step("inference")), notify=notify)

    assert [r.ok for r in results] == [False, True]
    assert "SystemExit" in results[0].detail
    assert len(notify.messages) == 1  # tổng kết vẫn gửi


async def test_tat_ca_ok_notify_tong_ket_xanh():
    notify = _Notify()
    results = await run_daily_pipeline(
        steps=(_ok_step("prices", "30 mã"), _ok_step("inference", "30 predictions")),
        notify=notify,
    )
    assert all(r.ok for r in results)  # caller (script) tính exit 0 từ đây
    msg = notify.messages[0]
    assert msg.startswith("✅")
    assert "30 mã" in msg and "30 predictions" in msg


async def test_notify_loi_khong_lam_pipeline_raise():
    results = await run_daily_pipeline(steps=(_ok_step("prices"),), notify=_Notify(raise_exc=True))
    assert all(r.ok for r in results)  # không raise, kết quả vẫn trả về


def test_build_summary_warning_crawl_0_tin():
    msg = build_summary(
        [
            StepResult("prices", True, "30 mã"),
            StepResult("news", True, "0 bài mới — kiểm tra crawler/nguồn", warning=True),
        ]
    )
    assert msg.startswith("⚠️")  # có warning, không fail
    assert "⚠️ news" in msg


def test_build_summary_fail_thang_warning():
    msg = build_summary(
        [
            StepResult("news", True, "0 bài", warning=True),
            StepResult("inference", False, "RuntimeError: x"),
        ]
    )
    assert msg.startswith("❌")  # fail trội hơn warning
    assert "1 bước FAIL" in msg


@pytest.mark.parametrize("flag", [True, False])
async def test_lifespan_scheduler_theo_flag(flag, monkeypatch):
    """ENABLE_SCHEDULER bật → scheduler start trong lifespan; tắt (default) → không."""
    import main as main_module
    from config import settings

    started: list[bool] = []

    class _FakeScheduler:
        def start(self):
            started.append(True)

        def shutdown(self, wait=False):
            pass

    monkeypatch.setattr(settings, "enable_scheduler", flag)
    monkeypatch.setattr("services.scheduler.create_scheduler", lambda: _FakeScheduler())

    async with main_module.lifespan(main_module.app):
        pass
    assert bool(started) is flag
