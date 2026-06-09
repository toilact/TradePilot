"""Test crawler: parser CafeF (HTML fixture) + persist idempotent + bất biến data governance."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from models.database import News, NewsStock
from services import crawler, stock_seed

FIXTURE = Path(__file__).parent / "fixtures" / "cafef_vcb.html"


def test_parse_cafef_extracts_fields():
    rows = crawler.parse_cafef(FIXTURE.read_text(encoding="utf-8"))
    assert len(rows) == 6
    first = rows[0]
    assert (
        first["title"] == "Chứng khoán ngày 19/5: Nhóm dầu khí bị bán mạnh, VN-Index giảm 15 điểm"
    )
    assert first["url"].startswith("https://cafef.vn/")
    assert first["url"].endswith(".chn")
    assert first["source"] == "cafef"
    # published_at PHẢI có giờ (naive, giờ VN) — nền tảng chống leakage
    assert first["published_at"] == datetime(2026, 5, 19, 17, 36)


def test_parse_cafef_skips_malformed():
    a = 'class="box-category-link-title"'
    html = f"""
    <div class="box-category-item">
      <a href="/thieu-thoi-gian.chn" {a}>Thiếu thời gian</a>
    </div>
    <div class="box-category-item">
      <a href="/ok-188260519171847749.chn" title="Bài hợp lệ" {a}>Bài hợp lệ</a>
      <p class="time_cate"><span class="time">19/05/2026 09:00</span></p>
    </div>
    """
    rows = crawler.parse_cafef(html)
    assert len(rows) == 1  # bài thiếu time bị bỏ
    assert rows[0]["title"] == "Bài hợp lệ"


def test_parse_cafef_empty():
    assert crawler.parse_cafef("<div></div>") == []


async def test_persist_idempotent_and_content_null(session_factory, monkeypatch):
    monkeypatch.setattr(crawler, "SessionLocal", session_factory)
    monkeypatch.setattr(stock_seed, "SessionLocal", session_factory)
    monkeypatch.setattr(
        stock_seed,
        "_fetch_metadata",
        lambda symbol, source="vci": {
            "symbol": symbol.upper(),
            "name": "Vietcombank",
            "exchange": "HOSE",
            "sector": None,
        },
    )
    stock_id = await stock_seed.seed_stock("VCB")
    rows = crawler.parse_cafef(FIXTURE.read_text(encoding="utf-8"))

    n1 = await crawler._persist(rows, stock_id)
    assert n1 == 6  # lần đầu: tất cả là bài mới

    n2 = await crawler._persist(rows, stock_id)
    assert n2 == 0  # chạy lại: không có bài mới (skip theo url)

    async with session_factory() as s:
        n_news = (await s.execute(select(func.count()).select_from(News))).scalar_one()
        n_link = (await s.execute(select(func.count()).select_from(NewsStock))).scalar_one()
        contents = (await s.execute(select(News.content))).scalars().all()

    assert n_news == 6  # idempotent — không nhân đôi
    assert n_link == 6  # mỗi bài map đúng 1 lần tới VCB
    # BẤT BIẾN: không tái xuất bản nguyên văn → content luôn None
    assert all(c is None for c in contents)


async def test_persist_maps_multiple_stocks_to_same_article(session_factory, monkeypatch):
    """1 bài → nhiều mã: cùng url, 2 stock_id → 1 news + 2 news_stocks."""
    monkeypatch.setattr(crawler, "SessionLocal", session_factory)
    monkeypatch.setattr(stock_seed, "SessionLocal", session_factory)
    monkeypatch.setattr(
        stock_seed,
        "_fetch_metadata",
        lambda symbol, source="vci": {
            "symbol": symbol.upper(),
            "name": symbol.upper(),
            "exchange": "HOSE",
            "sector": None,
        },
    )
    vcb = await stock_seed.seed_stock("VCB")
    ctg = await stock_seed.seed_stock("CTG")
    rows = [
        {
            "title": "Tin chung ngành ngân hàng",
            "url": "https://cafef.vn/tin-chung-188260519171847749.chn",
            "published_at": datetime(2026, 5, 19, 10, 0),
            "source": "cafef",
        }
    ]
    await crawler._persist(rows, vcb)
    await crawler._persist(rows, ctg)

    async with session_factory() as s:
        n_news = (await s.execute(select(func.count()).select_from(News))).scalar_one()
        n_link = (await s.execute(select(func.count()).select_from(NewsStock))).scalar_one()

    assert n_news == 1  # cùng url → 1 bài
    assert n_link == 2  # map tới cả 2 mã
