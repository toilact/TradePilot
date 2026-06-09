import type { PricePoint } from "@/lib/types";

// Chart giá lớn (SVG thuần) + overlay sentiment dưới chân. Responsive qua viewBox.
export function PriceChart({ data }: { data: PricePoint[] }) {
  const W = 760;
  const H = 260;
  const padB = 48; // chừa chỗ cho dải sentiment
  const closes = data.map((d) => d.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const span = max - min || 1;
  const stepX = W / (data.length - 1);
  const up = closes[closes.length - 1] >= closes[0];
  const color = up ? "#34d399" : "#fb7185";

  const pts = data.map((d, i) => {
    const x = i * stepX;
    const y = (H - padB) - ((d.close - min) / span) * (H - padB - 12) - 6;
    return [x, y] as const;
  });
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${W} ${H - padB} L0 ${H - padB} Z`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="pc-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.22" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* lưới ngang mờ */}
      {[0.25, 0.5, 0.75].map((g) => (
        <line
          key={g}
          x1="0"
          x2={W}
          y1={(H - padB) * g}
          y2={(H - padB) * g}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="1"
        />
      ))}

      <path d={area} fill="url(#pc-fill)" />
      <path d={line} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

      {/* dải sentiment dưới chân: cột xanh (dương) / đỏ (âm) */}
      {data.map((d, i) => {
        const h = Math.abs(d.sentiment) * 16;
        const x = i * stepX;
        const c = d.sentiment >= 0 ? "#34d399" : "#fb7185";
        return (
          <rect
            key={i}
            x={x - 1.5}
            width="3"
            y={H - 20 - (d.sentiment >= 0 ? h : 0)}
            height={Math.max(1, h)}
            rx="1"
            fill={c}
            opacity="0.5"
          />
        );
      })}
    </svg>
  );
}
