// Sparkline SVG thuần — không dùng lib chart nặng. Tô gradient theo xu hướng.
export function Sparkline({
  values,
  up = true,
  width = 120,
  height = 36,
  strokeWidth = 1.5,
}: {
  values: number[];
  up?: boolean;
  width?: number;
  height?: number;
  strokeWidth?: number;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = width / (values.length - 1);
  const pts = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / span) * (height - strokeWidth * 2) - strokeWidth;
    return [x, y] as const;
  });
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${d} L${width} ${height} L0 ${height} Z`;
  const color = up ? "#34d399" : "#fb7185";
  const id = `spark-${up ? "u" : "d"}-${values.length}`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
