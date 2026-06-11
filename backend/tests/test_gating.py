"""Test confidence-gating ở tầng đọc API (M3 — ADR 0002).

Kiểm: field gating mới (probs/isActionable/threshold/display), ưu tiên non-stub khi
trùng ngày, accuracy coverage/precision chấm theo đúng version hiện hành.
"""

from datetime import date

from models.database import ActualResult, Prediction, PriceHistory, Stock
from services import read_api
from services.read_api import DISPLAY_NO_SIGNAL


def _pred(sid: int, d: date, label: str, ver: str, **kw) -> Prediction:
    defaults = dict(
        stock_id=sid,
        prediction_date=d,
        target_date=date(d.year, d.month, d.day + 1),
        label=label,
        confidence=kw.pop("confidence", 0.5),
        model_version=ver,
    )
    return Prediction(**defaults, **kw)


async def _seed_stock(s, symbol: str = "VCB") -> int:
    stock = Stock(symbol=symbol, name=symbol, exchange="HSX", sector="Ngân hàng")
    s.add(stock)
    await s.flush()
    s.add(
        PriceHistory(
            stock_id=stock.id,
            date=date(2026, 6, 9),
            open=60,
            high=61,
            low=59,
            close=60.0,
            volume=1000,
        )
    )
    return stock.id


async def test_actionable_prediction_returns_full_gating_fields(session_factory):
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(
            _pred(
                sid,
                date(2026, 6, 9),
                "tang",
                "lgbm_v4",
                confidence=0.71,
                prob_tang=0.71,
                prob_giam=0.09,
                prob_di_ngang=0.20,
                is_actionable=True,
                threshold=0.60,
            )
        )
        await s.commit()
        p = await read_api.get_prediction(s, "VCB")
    assert p["display"] == "tang"  # actionable → display = label
    assert p["isActionable"] is True
    assert p["threshold"] == 0.60
    assert (p["probTang"], p["probGiam"], p["probDiNgang"]) == (0.71, 0.09, 0.20)


async def test_not_actionable_displays_no_signal_but_keeps_probs(session_factory):
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(
            _pred(
                sid,
                date(2026, 6, 9),
                "giam",
                "lgbm_v4",
                confidence=0.45,
                prob_tang=0.25,
                prob_giam=0.45,
                prob_di_ngang=0.30,
                is_actionable=False,
                threshold=0.60,
            )
        )
        await s.commit()
        p = await read_api.get_prediction(s, "VCB")
    assert p["display"] == DISPLAY_NO_SIGNAL
    assert p["label"] == "giam"  # vẫn trả để trang chi tiết minh bạch prob bar
    assert p["probGiam"] == 0.45


async def test_stub_fields_are_null_and_not_actionable(session_factory):
    """Bản ghi stub_v0 cũ (trước migration): prob/threshold NULL, is_actionable=false."""
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(_pred(sid, date(2026, 6, 9), "tang", "stub_v0", confidence=0.5))
        await s.commit()
        p = await read_api.get_prediction(s, "VCB")
    assert p["display"] == DISPLAY_NO_SIGNAL
    assert p["probTang"] is None and p["threshold"] is None
    assert p["isActionable"] is False


async def test_same_day_prefers_non_stub(session_factory):
    """Cùng prediction_date có cả stub_v0 lẫn lgbm_v4 → API trả lgbm_v4 (cả single + list)."""
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(_pred(sid, date(2026, 6, 9), "di_ngang", "stub_v0"))
        s.add(
            _pred(
                sid,
                date(2026, 6, 9),
                "tang",
                "lgbm_v4",
                confidence=0.66,
                prob_tang=0.66,
                prob_giam=0.14,
                prob_di_ngang=0.20,
                is_actionable=True,
                threshold=0.60,
            )
        )
        await s.commit()
        single = await read_api.get_prediction(s, "VCB")
        lst = await read_api.list_predictions(s)
    assert single["modelVersion"] == "lgbm_v4"
    assert lst[0]["modelVersion"] == "lgbm_v4"
    assert lst[0]["display"] == "tang"


async def test_newer_date_wins_even_if_stub(session_factory):
    """Ưu tiên non-stub CHỈ khi trùng ngày — ngày mới hơn luôn thắng."""
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(
            _pred(
                sid,
                date(2026, 6, 8),
                "tang",
                "lgbm_v4",
                is_actionable=True,
                threshold=0.60,
                prob_tang=0.7,
                prob_giam=0.1,
                prob_di_ngang=0.2,
            )
        )
        s.add(_pred(sid, date(2026, 6, 9), "giam", "stub_v0"))
        await s.commit()
        p = await read_api.get_prediction(s, "VCB")
    assert p["modelVersion"] == "stub_v0"


async def test_accuracy_coverage_precision_on_current_version(session_factory):
    """3 pred lgbm_v4 (2 actionable) + 1 stub_v0 cũ; actual đủ → coverage 2/3,
    precision chỉ chấm trên 2 actionable (1 đúng 1 sai = 0.5), stub không lẫn vào."""
    async with session_factory() as s:
        sid = await _seed_stock(s, "VCB")
        sid2 = await _seed_stock(s, "FPT")
        sid3 = await _seed_stock(s, "HPG")
        d = date(2026, 6, 9)
        # stub cũ ngày trước — không phải version hiện hành
        s.add(_pred(sid, date(2026, 6, 8), "tang", "stub_v0"))
        gating = dict(threshold=0.60, prob_tang=0.5, prob_giam=0.2, prob_di_ngang=0.3)
        s.add(_pred(sid, d, "tang", "lgbm_v4", is_actionable=True, **gating))  # đúng
        s.add(_pred(sid2, d, "giam", "lgbm_v4", is_actionable=True, **gating))  # sai
        s.add(_pred(sid3, d, "tang", "lgbm_v4", is_actionable=False, **gating))  # không tính
        s.add_all(
            [
                ActualResult(stock_id=sid, date=d, label="tang"),
                ActualResult(stock_id=sid2, date=d, label="tang"),
                ActualResult(stock_id=sid3, date=d, label="tang"),
            ]
        )
        await s.commit()
        acc = await read_api.get_accuracy(s)
    assert acc["modelVersion"] == "lgbm_v4"
    assert acc["coverage"] == round(2 / 3, 4)
    assert acc["precisionActionable"] == 0.5


async def test_accuracy_no_actuals_still_reports_coverage(session_factory):
    """Chưa có actual_results → vẫn trả coverage; precision None (chưa chấm được)."""
    async with session_factory() as s:
        sid = await _seed_stock(s)
        s.add(
            _pred(
                sid,
                date(2026, 6, 9),
                "tang",
                "lgbm_v4",
                is_actionable=True,
                threshold=0.60,
                prob_tang=0.7,
                prob_giam=0.1,
                prob_di_ngang=0.2,
            )
        )
        await s.commit()
        acc = await read_api.get_accuracy(s)
    assert acc["coverage"] == 1.0
    assert acc["precisionActionable"] is None
    assert "detail" in acc  # nhánh stub-aware giữ nguyên
