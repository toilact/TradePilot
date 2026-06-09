import type { Label } from "@/lib/types";

const MAP: Record<Label, { text: string; cls: string; glyph: string }> = {
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
};

export function PredictionBadge({
  label,
  size = "md",
}: {
  label: Label;
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
