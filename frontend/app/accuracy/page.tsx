import { getAccuracy } from "@/lib/api";
import { Reveal } from "@/components/reveal";
import { StatCard } from "@/components/stat-card";
import { Disclaimer } from "@/components/disclaimer";
import { PredictionBadge } from "@/components/prediction-badge";
import type { Label } from "@/lib/types";

export default async function AccuracyPage() {
  const acc = await getAccuracy();
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  // chart đường accuracy
  const W = 760;
  const H = 180;
  const vals = acc.series.map((s) => s.accuracy);
  const min = Math.min(...vals) - 0.02;
  const max = Math.max(...vals) + 0.02;
  const span = max - min || 1;
  const stepX = W / (acc.series.length - 1);
  const pts = acc.series.map((s, i) => [i * stepX, H - ((s.accuracy - min) / span) * (H - 16) - 8] as const);
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${W} ${H} L0 ${H} Z`;

  const labels: { k: Label; name: string }[] = [
    { k: "tang", name: "Tăng" },
    { k: "giam", name: "Giảm" },
    { k: "di_ngang", name: "Đi ngang" },
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-32 md:px-8">
      <Reveal>
        <span className="eyebrow">Minh bạch · Model {acc.modelVersion}</span>
      </Reveal>
      <Reveal delay={80}>
        <h1 className="mt-6 max-w-2xl text-balance font-display text-5xl font-semibold leading-[1.08] tracking-tight text-white md:text-6xl">
          Mô hình đúng bao nhiêu?
        </h1>
      </Reveal>
      <Reveal delay={150}>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-white/65">
          Độ chính xác đo bằng cách so dự đoán với kết quả thực tế (close T+1). Baseline đoán ngẫu
          nhiên ≈ 33%. Trang này vừa để bạn đánh giá, vừa để chúng tôi giám sát drift của mô hình.
        </p>
      </Reveal>

      {/* Bento số liệu */}
      <section className="mt-14 grid grid-cols-1 gap-4 md:grid-cols-4 md:grid-rows-2">
        <Reveal className="md:col-span-2 md:row-span-2">
          <StatCard
            label="Độ chính xác tổng thể"
            value={pct(acc.overall)}
            sub="Trên toàn bộ lịch sử dự đoán"
            accent="#e8c39e"
            className="min-h-[220px]"
          />
        </Reveal>
        <Reveal delay={80} className="md:col-span-2">
          <StatCard label="30 phiên gần nhất" value={pct(acc.last30)} sub="Tín hiệu drift ngắn hạn" accent="#34d399" />
        </Reveal>
        {labels.map((l, i) => (
          <Reveal key={l.k} delay={120 + i * 60} className="md:col-span-2 lg:col-span-1">
            <div className="bezel-shell h-full">
              <div className="bezel-core flex h-full items-center justify-between gap-3 p-5">
                <PredictionBadge label={l.k} size="sm" />
                <span className="font-display text-2xl font-semibold text-white">
                  {pct(acc.byLabel[l.k])}
                </span>
              </div>
            </div>
          </Reveal>
        ))}
      </section>

      {/* Chart rolling accuracy */}
      <Reveal delay={120}>
        <div className="mt-6 bezel-shell">
          <div className="bezel-core p-6 md:p-8">
            <h2 className="mb-6 font-display text-lg font-semibold text-white">
              Độ chính xác cuộn · 30 phiên
            </h2>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
              <defs>
                <linearGradient id="acc-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e8c39e" stopOpacity="0.2" />
                  <stop offset="100%" stopColor="#e8c39e" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d={area} fill="url(#acc-fill)" />
              <path d={line} fill="none" stroke="#e8c39e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        </div>
      </Reveal>

      <div className="mt-12">
        <Disclaimer variant="banner" />
      </div>
    </div>
  );
}
