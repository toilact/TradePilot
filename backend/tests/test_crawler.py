"""Test crawler: parser CafeF (HTML fixture) + persist idempotent + bất biến data governance."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from models.database import News, NewsStock
from services import crawler, stock_seed

FIXTURE = Path(__file__).parent / "fixtures" / "cafef_vcb.html"
RSS_FIXTURE = Path(__file__).parent / "fixtures" / "cafef_rss.xml"


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


# ── RSS ───────────────────────────────────────────────────────────────────────


def test_parse_cafef_rss_extracts_fields():
    rows = crawler.parse_cafef_rss(RSS_FIXTURE.read_text(encoding="utf-8"))
    assert len(rows) == 6  # 1 item tổng hợp + 5 item thật
    first = rows[0]
    assert first["title"].startswith("VCB và CTG")
    assert first["url"].startswith("https://cafef.vn/") and first["url"].endswith(".chn")
    assert first["source"] == "cafef"
    # pubDate '+0700' → naive giờ VN (18:00, không lệch timezone)
    assert first["published_at"] == datetime(2026, 6, 9, 18, 0)


def test_parse_cafef_rss_invalid_xml():
    assert crawler.parse_cafef_rss("<rss><broken") == []


def test_parse_rss_time_converts_to_vn_naive():
    # +0000 09:00 UTC → 16:00 giờ VN, naive
    assert crawler._parse_rss_time("Tue, 09 Jun 2026 09:00:00 +0000") == datetime(2026, 6, 9, 16, 0)
    assert crawler._parse_rss_time(None) is None
    assert crawler._parse_rss_time("rác") is None


def test_match_stocks_symbol_and_name_multi():
    lookup = {"VCB": 1, "vietcombank": 1, "CTG": 2, "PET": 3, "petrosetco": 3}
    # title nhắc cả VCB (symbol), Vietcombank (tên→cùng mã 1), CTG (symbol)
    ids = crawler.match_stocks("VCB và CTG dẫn dắt nhóm ngân hàng, Vietcombank lập đỉnh", lookup)
    assert set(ids) == {1, 2}  # mã 1 chỉ xuất hiện 1 lần dù khớp 2 key


def test_match_stocks_token_boundary_and_no_match():
    lookup = {"PET": 3, "petrosetco": 3, "VCB": 1}
    # 'PETROVN' KHÔNG được khớp PET (token boundary); nhưng tên 'petrosetco' thì có
    assert crawler.match_stocks("Petrosetco tăng trần", lookup) == [3]
    assert crawler.match_stocks("PETROVN niêm yết", lookup) == []  # PET không dính chuỗi con
    assert crawler.match_stocks("Tin vĩ mô không có mã", lookup) == []


async def test_crawl_rss_maps_and_persists(session_factory, monkeypatch):
    monkeypatch.setattr(crawler, "SessionLocal", session_factory)
    monkeypatch.setattr(stock_seed, "SessionLocal", session_factory)
    monkeypatch.setattr(
        stock_seed,
        "_fetch_metadata",
        lambda symbol, source="vci": {
            "symbol": symbol.upper(),
            "name": "Vietcombank" if symbol.upper() == "VCB" else symbol.upper(),
            "exchange": "HOSE",
            "sector": None,
        },
    )
    await stock_seed.seed_stock("VCB")
    await stock_seed.seed_stock("CTG")
    # _fetch trả fixture (không gọi mạng); chỉ 1 feed để đếm rõ
    monkeypatch.setattr(
        crawler, "_fetch", lambda url: _async_return(RSS_FIXTURE.read_text(encoding="utf-8"))
    )

    new1 = await crawler.crawl_rss(feeds=("fake-feed",))
    assert new1 == 1  # chỉ bài "VCB và CTG..." khớp mã đã seed

    new2 = await crawler.crawl_rss(feeds=("fake-feed",))
    assert new2 == 0  # idempotent

    async with session_factory() as s:
        n_news = (await s.execute(select(func.count()).select_from(News))).scalar_one()
        n_link = (await s.execute(select(func.count()).select_from(NewsStock))).scalar_one()
        contents = (await s.execute(select(News.content))).scalars().all()

    assert n_news == 1  # 1 bài khớp
    assert n_link == 2  # map tới VCB + CTG
    assert all(c is None for c in contents)  # bất biến: content NULL


async def _async_return(value):
    return value
