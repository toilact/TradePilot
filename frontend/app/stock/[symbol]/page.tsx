import Link from "next/link";
import { notFound } from "next/navigation";
import { getHistory, getNews, getPrediction } from "@/lib/api";
import { MOCK_PREDICTIONS } from "@/lib/mock";
import { Reveal } from "@/components/reveal";
import { PredictionBadge } from "@/components/prediction-badge";
import { PriceChart } from "@/components/price-chart";
import { StatCard } from "@/components/stat-card";
import { Disclaimer } from "@/components/disclaimer";
import { changeColor, fmtConfidence, fmtPct, fmtPrice } from "@/lib/format";

export function generateStaticParams() {
  return MOCK_PREDICTIONS.map((p) => ({ symbol: p.symbol }));
}

export default async function StockPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = await params;
  const p = await getPrediction(symbol);
  if (!p) notFound();
  const [history, news] = await Promise.all([getHistory(symbol), getNews(symbol)]);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-32 md:px-8">
      {/* breadcrumb */}
      <Reveal>
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-white/60 transition-colors hover:text-white"
        >
          <span aria-hidden>←</span> Tất cả dự đoán
        </Link>
      </Reveal>

      {/* Header mã */}
      <Reveal delay={60}>
        <div className="mt-8 flex flex-wrap items-end justify-between gap-6">
          <div className="flex items-center gap-5">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] font-display text-lg font-semibold text-white">
              {p.symbol.slice(0, 3)}
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="font-display text-4xl font-semibold tracking-tight text-white">
                  {p.symbol}
                </h1>
                <span className="text-[11px] uppercase tracking-wider text-white/50">
                  {p.exchange} · {p.sector}
                </span>
              </div>
              <p className="mt-1 text-white/65">{p.name}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-3xl text-white">{fmtPrice(p.close)}</div>
            <div className={`text-sm ${changeColor(p.changePct)}`}>{fmtPct(p.changePct)} phiên gần nhất</div>
          </div>
        </div>
      </Reveal>

      {/* Dự đoán nổi bật */}
      <Reveal delay={120}>
        <div className="mt-10 bezel-shell">
          <div className="bezel-core flex flex-col gap-6 p-7 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-5">
              <span className="text-[11px] uppercase tracking-[0.18em] text-white/55">
                Dự đoán T+1
              </span>
              <PredictionBadge label={p.label} size="lg" />
            </div>
            <div className="flex items-center gap-8">
              <Metric label="Độ tin cậy" value={fmtConfidence(p.confidence)} />
              <Metric
                label="Sentiment"
                value={p.sentiment.toFixed(2)}
                accent={p.sentiment >= 0 ? "#34d399" : "#fb7185"}
              />
              <Metric label="Model" value={p.modelVersion} />
            </div>
          </div>
        </div>
      </Reveal>

      {/* Chart giá + sentiment */}
      <Reveal delay={160}>
        <div className="mt-6 bezel-shell">
          <div className="bezel-core p-6 md:p-8">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="font-display text-lg font-semibold text-white">Giá & sentiment · 60 phiên</h2>
              <div className="flex items-center gap-4 text-xs text-white/60">
                <Legend color="#34d399" label="Giá" />
                <Legend color="#8b5cf6" label="Sentiment" />
              </div>
            </div>
            <PriceChart data={history} />
          </div>
        </div>
      </Reveal>

      {/* Tin tức liên quan */}
      <section className="mt-16">
        <Reveal>
          <h2 className="mb-6 font-display text-2xl font-semibold tracking-tight text-white">
            Tin tức liên quan
          </h2>
        </Reveal>
        <div className="grid gap-3 md:grid-cols-2">
          {news.map((n, i) => (
            <Reveal key={i} delay={i * 60}>
              <a href={n.url} className="group block bezel-shell">
                <div className="bezel-core flex items-start gap-4 p-5">
                  <span
                    className="mt-1 h-2 w-2 shrink-0 rounded-full"
                    style={{ background: n.sentiment >= 0 ? "#34d399" : "#fb7185" }}
                  />
                  <div>
                    <p className="text-sm leading-snug text-white/85 transition-colors group-hover:text-white">
                      {n.title}
                    </p>
                    <p className="mt-2 text-xs text-white/55">
                      {n.source} · {n.publishedAt}
                    </p>
                  </div>
                </div>
              </a>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Lịch sử dự đoán (placeholder) */}
      <section className="mt-16 grid gap-4 md:grid-cols-3">
        <Reveal>
          <StatCard label="Đúng / 30 phiên" value="—" sub="Sẽ có khi tích lũy kết quả thực tế" />
        </Reveal>
        <Reveal delay={80}>
          <StatCard label="Chuỗi đúng gần nhất" value="—" sub="Đang chờ dữ liệu" />
        </Reveal>
        <Reveal delay={140}>
          <StatCard label="Số tin 7 ngày" value={String(news.length)} sub="Nguồn CafeF + FireAnt" />
        </Reveal>
      </section>

      <div className="mt-12">
        <Disclaimer variant="banner" />
      </div>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.14em] text-white/50">{label}</p>
      <p className="mt-1 font-display text-xl font-semibold" style={accent ? { color: accent } : undefined}>
        {value}
      </p>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
