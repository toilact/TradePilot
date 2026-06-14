import Link from "next/link";
import type { Prediction } from "@/lib/types";
import { PredictionBadge } from "./prediction-badge";
import { Sparkline } from "./sparkline";
import { WatchlistButton } from "./watchlist-button";
import { mockPriceSeries } from "@/lib/mock";
import { changeColor, fmtConfidence, fmtPct, fmtPrice } from "@/lib/format";

export function PredictionRow({ p }: { p: Prediction }) {
  const series = mockPriceSeries(p.symbol).map((d) => d.close);
  return (
    <div className="relative">
      {/* Nút theo dõi (M9) — sibling của Link (KHÔNG nhúng trong <a>) để HTML hợp lệ + không điều hướng */}
      <WatchlistButton
        symbol={p.symbol}
        className="absolute right-3 top-3 z-10"
      />
      <Link
        href={`/stock/${p.symbol}`}
        className="group grid grid-cols-[1fr_auto] items-center gap-4 rounded-[calc(2rem-0.75rem)] px-5 py-4 pr-12 transition-colors duration-300 ease-fluid hover:bg-white/[0.03] md:grid-cols-[1.4fr_1fr_1fr_1.2fr_1fr] md:gap-6 md:pr-5"
      >
      {/* Mã + tên */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] font-display text-xs font-semibold tracking-tight text-white/80">
          {p.symbol.slice(0, 3)}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-display text-sm font-semibold tracking-tight text-white">
              {p.symbol}
            </span>
            <span className="hidden text-[10px] uppercase tracking-wider text-white/45 sm:inline">
              {p.exchange}
            </span>
          </div>
          <p className="truncate text-xs text-white/60">{p.name}</p>
        </div>
      </div>

      {/* Giá + thay đổi */}
      <div className="hidden text-right md:block">
        <div className="font-mono text-sm text-white/85">{fmtPrice(p.close)}</div>
        <div className={`text-xs ${changeColor(p.changePct)}`}>{fmtPct(p.changePct)}</div>
      </div>

      {/* Sparkline */}
      <div className="hidden items-center justify-center md:flex">
        <Sparkline values={series} up={p.changePct >= 0} />
      </div>

      {/* Dự đoán + confidence — badge theo `display` (backend quyết định gating) */}
      <div className="hidden items-center gap-3 md:flex">
        <PredictionBadge label={p.display} />
        {p.isActionable && <ConfidenceMeter value={p.confidence} />}
      </div>

      {/* Mobile: badge + arrow */}
      <div className="flex items-center justify-end gap-3">
        <div className="md:hidden">
          <PredictionBadge label={p.display} size="sm" />
        </div>
        <span className="hidden text-xs text-white/55 md:inline">
          {p.isActionable ? `${fmtConfidence(p.confidence)} tin cậy` : "dưới ngưỡng"}
        </span>
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-white/[0.04] text-white/55 transition-all duration-500 ease-spring group-hover:translate-x-0.5 group-hover:bg-white/10 group-hover:text-white">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M9 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </div>
      </Link>
    </div>
  );
}

function ConfidenceMeter({ value }: { value: number }) {
  return (
    <div className="hidden h-1 w-14 overflow-hidden rounded-full bg-white/10 lg:block">
      <div
        className="h-full rounded-full bg-gradient-to-r from-white/40 to-white/80"
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}
