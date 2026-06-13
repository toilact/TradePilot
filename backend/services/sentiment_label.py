"""Auto-label sentiment cho title tin bằng LLM (Gemini, fallback OpenAI) — data train PhoBERT.

CHỈ dùng để gán nhãn data train (ml/), KHÔNG ghi sentiment_score vào DB (đó là việc của
PhoBERT sau khi fine-tune). Governance: người review mẫu sai sau khi auto-label.

Nhãn: 3 lớp pos / neu / neg theo rubric (góc tác động giá mã T+1, ngưỡng bảo thủ → mặc định neu;
chi tiết ở `ml/data/LABELING_RUBRIC.md` — tài liệu nội bộ gitignored). Prompt dưới khớp rubric đó.
Provider trừu tượng hoá (_Provider): Gemini đa key xoay vòng khi hết quota/key chết; hết đường
→ OpenAI (autolabel script tự fallback). SDK import trễ theo provider (group `pipeline`).

⚠️ SDK Gemini: `google-generativeai` đã deprecated (còn chạy); model `gemini-1.5-flash` đã
RETIRE → dùng `gemini-2.5-flash`. OpenAI dùng `gpt-4o-mini`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

LABELS = ("pos", "neu", "neg")
# Nhãn cho batch thất bại vĩnh viễn sau retry — KHÔNG đầu độc data bằng 'neu' thầm lặng.
# Bước kiểm định + notebook fine-tune loại các dòng ERROR (chỉ giữ pos/neu/neg).
ERROR_LABEL = "ERROR"
MODEL_NAME = "gemini-2.5-flash"  # Gemini mặc định (free tier ~20 req/NGÀY/key)
OPENAI_MODEL = "gpt-4o-mini"  # fallback khi Gemini hết quota — quota riêng, rẻ

_PROMPT = """Bạn là chuyên gia phân tích tin tức chứng khoán Việt Nam. Phân loại CẢM XÚC
(sentiment) của mỗi tiêu đề theo **tác động lên GIÁ CỔ PHIẾU của mã được nhắc, ngắn hạn (T+1)**:
- "pos": tín hiệu định hướng giá LÊN tường minh (lãi/kỷ lục, tăng/trần, mua ròng, chia cổ tức
  tiền/cổ phiếu, nâng hạng, hưởng lợi rõ).
- "neg": tín hiệu định hướng giá XUỐNG tường minh (lỗ, giảm/sàn, bán ròng/bán tháo, cổ đông lớn
  thoái vốn, cảnh báo/rủi ro, áp lực cung).
- "neu": KHÔNG có tín hiệu giá tường minh — tin thủ tục, huy động vốn/phát hành trái phiếu thường
  lệ, bổ nhiệm/miễn nhiệm nhân sự, sự kiện/PR, clickbait không lộ chiều, chiều mơ hồ/yếu.

QUY TẮC NGƯỠNG: **mặc định "neu"**; chỉ gán pos/neg khi tiêu đề có dấu hiệu định hướng giá TƯỜNG
MINH. Tin nghiêng nhẹ một chiều nhưng tác động giá không rõ → neu (KHÔNG ép chiều).
MULTI-MÃ: gán theo mã chính của tiêu đề; nếu đối xứng 2 chiều ("X tăng, Y giảm") → neu.

Ví dụ:
- "Vietcombank lãi kỷ lục quý 2" → pos
- "VRE chi 2.300 tỷ chia cổ tức tiền mặt" → pos
- "Khối ngoại bán ròng 1.500 tỷ, cổ phiếu nào bị xả mạnh" → neg
- "Cổ phiếu BIDV bất ngờ giảm sàn" → neg
- "VPBank phát hành 1.000 tỷ trái phiếu" → neu (huy động vốn thường lệ)
- "Lãnh đạo FPT được mua ESOP thấp hơn thị giá 90%" → neu (không định hướng giá T+1)
- "Becamex miễn nhiệm Tổng giám đốc" → neu (nhân sự, trừ khi nêu bê bối)

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
    # Lấy mảng JSON đầu tiên (chịu được rào ```json``` lẫn giải thích/ví dụ LLM lỡ thêm trước).
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    labels = json.loads(text)
    if not isinstance(labels, list) or len(labels) != n:
        got = len(labels) if isinstance(labels, list) else "?"
        raise ValueError(f"LLM trả {got} nhãn, cần {n}")
    return [lbl if lbl in LABELS else "neu" for lbl in labels]


def label_to_score(label: str) -> float:
    """Map nhãn → điểm [-1,1] (dùng khi score_text thật chấm: pos=+1, neu=0, neg=-1)."""
    return {"pos": 1.0, "neg": -1.0}.get(label, 0.0)


# Hành động khi 1 batch lỗi: ROTATE (đổi key/skip, retry ngay) · RETRY (tạm thời, backoff) ·
# FATAL (nhà cung cấp hết đường — dừng sớm, phần còn lại gán ERROR).
ROTATE, RETRY, FATAL = "rotate", "retry", "fatal"


class _Provider(Protocol):
    """Một nhà cung cấp LLM gán nhãn 1 batch; tự quyết cách xử lý lỗi."""

    async def label_batch(self, batch: list[str]) -> list[str]: ...
    def handle_error(self, exc: Exception) -> str: ...  # ROTATE | RETRY | FATAL


class _GeminiProvider:
    """Gemini đa key: hết quota (429) hoặc key bị chặn (403 denied) → bỏ key, xoay key kế."""

    def __init__(self, model_name: str, keys: list[str]) -> None:
        import google.generativeai as genai

        self._genai = genai
        self._model_name = model_name
        self._keys = keys
        self._ki = 0
        self._dead: set[int] = set()
        self._build()

    def _build(self) -> None:
        self._genai.configure(api_key=self._keys[self._ki])
        self._model = self._genai.GenerativeModel(self._model_name)

    async def label_batch(self, batch: list[str]) -> list[str]:
        resp = await self._model.generate_content_async(build_prompt(batch))
        return parse_labels(resp.text, len(batch))

    def handle_error(self, exc: Exception) -> str:
        s = str(exc).lower()
        key_dead = (
            "429" in s
            or "quota" in s
            or "resourceexhausted" in type(exc).__name__.lower()
            or "403" in s
            or "denied" in s
            or "permission" in s
        )
        if not key_dead:
            return RETRY
        self._dead.add(self._ki)
        if len(self._dead) >= len(self._keys):
            return FATAL
        while self._ki in self._dead:
            self._ki = (self._ki + 1) % len(self._keys)
        self._build()
        logger.warning("Gemini key chết → xoay sang key #%d", self._ki + 1)
        return ROTATE


class _OpenAIProvider:
    """OpenAI 1 key: rate-limit → backoff; lỗi auth/hết tiền → FATAL."""

    def __init__(self, model_name: str, api_key: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name

    async def label_batch(self, batch: list[str]) -> list[str]:
        resp = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": build_prompt(batch)}],
            temperature=0,
        )
        return parse_labels(resp.choices[0].message.content or "", len(batch))

    def handle_error(self, exc: Exception) -> str:
        s = str(exc).lower()
        if any(k in s for k in ("401", "invalid_api_key", "insufficient_quota", "billing")):
            return FATAL
        return RETRY  # rate-limit / lỗi mạng → backoff


def _make_provider(provider: str, model_name: str | None, api_keys: list[str] | None) -> _Provider:
    from config import settings

    if provider == "gemini":
        keys = api_keys if api_keys is not None else settings.gemini_api_keys
        if not keys:
            raise RuntimeError("Thiếu GEMINI_API_KEY trong .env — không auto-label được.")
        return _GeminiProvider(model_name or MODEL_NAME, keys)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("Thiếu OPENAI_API_KEY trong .env — không dùng fallback được.")
        return _OpenAIProvider(model_name or OPENAI_MODEL, settings.openai_api_key)
    raise ValueError(f"provider lạ: {provider!r} (chỉ 'gemini' | 'openai')")


async def label_titles(
    titles: list[str],
    batch_size: int = 20,
    max_retries: int = 3,
    pace_seconds: float = 4.0,
    model_name: str | None = None,
    api_keys: list[str] | None = None,
    provider: str = "gemini",
) -> list[str]:
    """Gán nhãn cho list title bằng LLM. Trả list nhãn CÙNG ĐỘ DÀI với `titles`.

    Chất lượng data > tốc độ: lỗi tạm thời → retry backoff (`max_retries`); batch FATAL/bí →
    gán ERROR_LABEL (KHÔNG 'neu' thầm lặng, tránh đầu độc data). `pace_seconds` giãn batch.
    `provider`: 'gemini' (đa key xoay vòng, free tier ~20 req/ngày/key) | 'openai' (fallback).
    Import SDK trễ (chỉ provider được chọn).
    """
    if not titles:
        return []

    prov = _make_provider(provider, model_name, api_keys)
    out: list[str] = []
    n_batches = (len(titles) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(titles), batch_size)):
        batch = titles[i : i + batch_size]
        labels: list[str] | None = None
        fatal = False
        tries = 0
        while labels is None and tries < max_retries:
            try:
                labels = await prov.label_batch(batch)
            except Exception as exc:
                action = prov.handle_error(exc)
                if action == FATAL:
                    logger.error(
                        "batch %d/%d: provider %s hết đường — dừng sớm", bi + 1, n_batches, provider
                    )
                    fatal = True
                    break
                if action == ROTATE:
                    continue  # không tính là 1 lượt retry
                tries += 1
                wait = pace_seconds * (2**tries)
                logger.warning(
                    "batch %d/%d lỗi tạm thời lần %d/%d (%s) — chờ %.0fs",
                    bi + 1,
                    n_batches,
                    tries,
                    max_retries,
                    exc,
                    wait,
                )
                if tries < max_retries:
                    await asyncio.sleep(wait)
        if labels is None:
            out.extend([ERROR_LABEL] * len(batch))
            if fatal:  # mọi batch sau cũng fail → điền ERROR phần còn lại + dừng
                remaining = len(titles) - (i + len(batch))
                if remaining > 0:
                    out.extend([ERROR_LABEL] * remaining)
                break
        else:
            out.extend(labels)
        if bi + 1 < n_batches:  # giãn cách giữa batch (trừ batch cuối)
            await asyncio.sleep(pace_seconds)
    return out
