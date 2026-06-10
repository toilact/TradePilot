"""Auto-label sentiment cho title tin bằng LLM (Gemini) — tạo data train PhoBERT.

CHỈ dùng để gán nhãn data train (ml/), KHÔNG ghi sentiment_score vào DB (đó là việc của
PhoBERT sau khi fine-tune). Governance: người review mẫu sai sau khi auto-label.

Nhãn: 3 lớp pos / neu / neg (tích cực / trung lập / tiêu cực) cho thị trường chứng khoán VN.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

LABELS = ("pos", "neu", "neg")

_PROMPT = """Bạn là chuyên gia phân tích tin tức chứng khoán Việt Nam. Phân loại CẢM XÚC
(sentiment) của mỗi tiêu đề tin theo góc nhìn NHÀ ĐẦU TƯ với mã/thị trường được nhắc:
- "pos": tích cực (giá/lợi nhuận/triển vọng tốt, mua ròng, tăng trưởng...)
- "neg": tiêu cực (giảm, bán mạnh, thua lỗ, rủi ro, áp lực...)
- "neu": trung lập (thông tin thuần, không rõ chiều, thủ tục/lịch sự kiện...)

Ví dụ:
- "Vietcombank lãi kỷ lục quý 2" → pos
- "Nhóm dầu khí bị bán mạnh, VN-Index giảm 15 điểm" → neg
- "FPT chốt danh sách cổ đông trả cổ tức" → neu

Trả về DUY NHẤT một mảng JSON các nhãn theo ĐÚNG thứ tự tiêu đề, vd: ["pos","neg","neu"].
Số phần tử phải bằng số tiêu đề. KHÔNG giải thích.

Danh sách tiêu đề (JSON):
{titles}
"""


def build_prompt(titles: list[str]) -> str:
    """Prompt vài-shot cho 1 batch title."""
    return _PROMPT.format(titles=json.dumps(titles, ensure_ascii=False))


def parse_labels(raw: str, n: int) -> list[str]:
    """Parse phản hồi LLM → list nhãn hợp lệ độ dài n.

    Chấp nhận text có rào ```json. Nhãn lạ → 'neu' (an toàn, không bịa chiều). Thiếu/thừa
    phần tử → raise (caller xử lý batch lỗi), tránh lệch alignment title↔nhãn.
    """
    text = raw.strip()
    if "```" in text:  # bóc rào code nếu có
        text = text.split("```")[1].removeprefix("json").strip()
    labels = json.loads(text)
    if not isinstance(labels, list) or len(labels) != n:
        got = len(labels) if isinstance(labels, list) else "?"
        raise ValueError(f"LLM trả {got} nhãn, cần {n}")
    return [lbl if lbl in LABELS else "neu" for lbl in labels]


def label_to_score(label: str) -> float:
    """Map nhãn → điểm [-1,1] (dùng khi score_text thật chấm: pos=+1, neu=0, neg=-1)."""
    return {"pos": 1.0, "neg": -1.0}.get(label, 0.0)


async def label_titles(titles: list[str], batch_size: int = 20) -> list[str]:
    """Gán nhãn cho list title bằng Gemini. Trả list nhãn cùng độ dài (lỗi batch → 'neu').

    Cần GEMINI_API_KEY trong .env. Import google-generativeai trễ (chỉ khi chạy thật).
    """
    if not titles:
        return []

    import google.generativeai as genai

    from config import settings

    if not settings.gemini_api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong .env — không auto-label được.")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    out: list[str] = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        try:
            resp = await model.generate_content_async(build_prompt(batch))
            out.extend(parse_labels(resp.text, len(batch)))
        except Exception as exc:  # batch lỗi → neu (an toàn), log để review
            logger.warning("batch %d-%d lỗi (%s) → gán 'neu'", i, i + len(batch), exc)
            out.extend(["neu"] * len(batch))
    return out
