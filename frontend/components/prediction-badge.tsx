import type { Display } from "@/lib/types";

// Trạng thái 4 "khong_du_tin_hieu" (M3 — confidence-gating): xám mờ trung tính,
// KHÔNG dùng gold (gold = brand, ADR 0001) và nhạt hơn flat (flat = data Đi ngang).
const MAP: Record<Display, { text: string; cls: string; glyph: string }> = {
  tang: {
    text: "Tăng",
    cls: "text-up border-up/30 bg-up/10",
    glyph: "↑",
  },
  giam: {
    text: "Giảm",
    cls: "text-down border-down/30 bg-down/10",
    glyph: "↓",
  },
  di_ngang: {
    text: "Đi ngang",
    cls: "text-flat border-flat/30 bg-flat/10",
    glyph: "→",
  },
  khong_du_tin_hieu: {
    text: "Không đủ tín hiệu",
    cls: "text-white/45 border-white/10 bg-white/[0.04]",
    glyph: "·",
  },
};

export function PredictionBadge({
  label,
  size = "md",
}: {
  label: Display;
  size?: "sm" | "md" | "lg";
}) {
  const m = MAP[label];
  const sizing =
    size === "lg"
      ? "px-4 py-1.5 text-sm"
      : size === "sm"
        ? "px-2.5 py-0.5 text-xs"
        : "px-3 py-1 text-[13px]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium tracking-tight ${m.cls} ${sizing}`}
    >
      <span aria-hidden className="font-display">
        {m.glyph}
      </span>
      {m.text}
    </span>
  );
}
