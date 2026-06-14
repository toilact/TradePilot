"""Ánh xạ tin vĩ mô/ngành → toàn bộ mã cùng ngành (M8 Pha 1a — tăng coverage tin).

Vấn đề: tin ngành/vĩ mô CafeF ("Ngân hàng Nhà nước hạ lãi suất", "Giá thép tăng") không chứa
mã cổ phiếu cụ thể → match_stocks (symbol/tên) bỏ sót → coverage tin thấp (~3.8% phiên-mã).
Giải pháp: từ điển keyword-NGÀNH; tiêu đề chứa keyword → map (N-N) tới MỌI mã VN30 cùng
`stocks.sector` (ICB level 2). 1 tin ngân hàng tốt/xấu → tác động cả 13 mã ngân hàng.

Keyword chọn theo hướng CHÍNH XÁC CAO: ưu tiên cụm nhiều từ ("lãi suất huy động", "giá thép")
để tránh false-positive; tránh cụm quá rộng ("chứng khoán" — gần như mọi tin thị trường có).

⚠️ Lưu ý phương pháp luận (xem plan M8): mapping này khiến 1 tin → N mã CÙNG NGÀY mang y hệt
điểm sentiment → có thể tạo artifact "nhớ ngày" (như MI index giả của lgbm_v4). Diagnostic
Pha 1 (null-test shuffle trong ngày + kiểm chứng 1-mã + test out-of-time) có nhiệm vụ phát hiện.

Key của SECTOR_KEYWORDS phải TRÙNG KHỚP giá trị `stocks.sector` để join ra stock_id.
"""

from __future__ import annotations

# {tên ngành (== stocks.sector) : cụm keyword lowercase}. Tiêu đề (lower) chứa 1 cụm → khớp ngành.
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Ngân hàng": (
        "ngân hàng",
        "nợ xấu",
        "lãi suất huy động",
        "lãi suất cho vay",
        "tăng trưởng tín dụng",
        "room tín dụng",
        "ngân hàng nhà nước",
        "nhnn",
    ),
    "Bất động sản": (
        "bất động sản",
        "địa ốc",
        "thị trường nhà đất",
        "phân khúc căn hộ",
        "đất nền",
    ),
    "Thực phẩm và đồ uống": (
        "thực phẩm",
        "đồ uống",
        "ngành sữa",
        "bia rượu",
        "giá đường",
    ),
    "Điện, nước & xăng dầu khí đốt": (
        "giá điện",
        "ngành điện",
        "thủy điện",
        "nhiệt điện",
        "điện than",
        "khí đốt",
    ),
    "Dầu khí": (
        "giá dầu",
        "dầu thô",
        "xăng dầu",
        "giá xăng",
        "opec",
    ),
    "Tài nguyên Cơ bản": (
        "giá thép",
        "ngành thép",
        "quặng sắt",
        "thép xây dựng",
        "hpg",
    ),
    "Dịch vụ tài chính": (
        "công ty chứng khoán",
        "ngành chứng khoán",
        "nghiệp vụ môi giới",
        "dư nợ margin",
    ),
    "Bán lẻ": (
        "ngành bán lẻ",
        "bán lẻ",
        "sức mua tiêu dùng",
    ),
    "Du lịch và Giải trí": (
        "hàng không",
        "ngành hàng không",
        "khách du lịch",
        "lượng khách quốc tế",
    ),
    "Hóa chất": (
        "giá cao su",
        "ngành cao su",
        "hóa chất",
    ),
    "Bảo hiểm": (
        "ngành bảo hiểm",
        "doanh thu phí bảo hiểm",
    ),
    "Công nghệ Thông tin": (
        "công nghệ thông tin",
        "ngành công nghệ",
        "chuyển đổi số",
    ),
}


def match_sector_stock_ids(title: str, sector_to_ids: dict[str, list[int]]) -> list[int]:
    """Tiêu đề chứa keyword ngành nào → trả MỌI stock_id thuộc ngành đó (union, loại trùng).

    `sector_to_ids`: map tên ngành (== stocks.sector) → list stock_id (dựng từ DB 1 lần).
    Giữ thứ tự xuất hiện ổn định. Tiêu đề không khớp ngành nào → [].
    """
    title_l = title.lower()
    found: dict[int, None] = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in title_l for kw in keywords):
            for sid in sector_to_ids.get(sector, ()):
                found[sid] = None
    return list(found)


async def load_sector_lookup() -> dict[str, list[int]]:
    """Dựng map ngành → [stock_id] từ bảng stocks (mã active có sector)."""
    from sqlalchemy import select

    from models.database import SessionLocal, Stock

    out: dict[str, list[int]] = {}
    async with SessionLocal() as session:
        rows = (
            await session.execute(select(Stock.id, Stock.sector).where(Stock.is_active.is_(True)))
        ).all()
    for sid, sector in rows:
        if sector:
            out.setdefault(sector, []).append(sid)
    return out
