"use client";

import { useState } from "react";
import type { Prediction } from "@/lib/types";
import { PredictionRow } from "./prediction-row";

// Bảng dự đoán + filter "chỉ mã có tín hiệu" (M3 — gating). Mặc định TẮT: hiện cả
// 30 mã (giá/thay đổi vẫn hữu ích), mã không tín hiệu mang badge xám.
export function PredictionTable({ preds }: { preds: Prediction[] }) {
  const [onlySignal, setOnlySignal] = useState(false);
  const shown = onlySignal ? preds.filter((p) => p.isActionable) : preds;

  return (
    <div>
      <div className="mb-4 flex justify-end">
        <button
          type="button"
          onClick={() => setOnlySignal((v) => !v)}
          aria-pressed={onlySignal}
          className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs transition-colors duration-300 ease-fluid ${
            onlySignal
              ? "border-gold/40 bg-gold/10 text-gold"
              : "border-white/10 bg-white/[0.04] text-white/60 hover:text-white"
          }`}
        >
          <span
            aria-hidden
            className={`h-1.5 w-1.5 rounded-full ${onlySignal ? "bg-gold" : "bg-white/30"}`}
          />
          Chỉ mã có tín hiệu ({preds.filter((p) => p.isActionable).length})
        </button>
      </div>

      <div className="bezel-shell">
        <div className="bezel-core overflow-hidden">
          {/* Header (desktop) */}
          <div className="hidden grid-cols-[1.4fr_1fr_1fr_1.2fr_1fr] gap-6 border-b border-white/5 px-5 py-3 text-[11px] uppercase tracking-[0.16em] text-white/50 md:grid">
            <span>Mã</span>
            <span className="text-right">Giá / Thay đổi</span>
            <span className="text-center">60 phiên</span>
            <span>Dự đoán T+1</span>
            <span className="text-right">Chi tiết</span>
          </div>
          <div className="divide-y divide-white/5">
            {shown.map((p) => (
              <PredictionRow key={p.symbol} p={p} />
            ))}
            {shown.length === 0 && (
              <p className="px-5 py-10 text-center text-sm text-white/50">
                Phiên này model không đủ tự tin với mã nào — tắt bộ lọc để xem toàn bộ.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
