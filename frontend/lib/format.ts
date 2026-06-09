export function fmtPrice(v: number): string {
  return v.toLocaleString("vi-VN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtPct(v: number): string {
  const s = v.toFixed(2);
  return `${v > 0 ? "+" : ""}${s}%`;
}

export function fmtConfidence(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export function changeColor(v: number): string {
  if (v > 0.01) return "text-up";
  if (v < -0.01) return "text-down";
  return "text-flat";
}
