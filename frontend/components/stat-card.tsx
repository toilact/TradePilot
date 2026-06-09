import type { ReactNode } from "react";

// Card thống kê dùng double-bezel.
export function StatCard({
  label,
  value,
  sub,
  accent,
  className = "",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
  className?: string;
}) {
  return (
    <div className={`bezel-shell h-full ${className}`}>
      <div className="bezel-core flex h-full flex-col justify-between gap-6 p-6">
        <p className="text-[11px] uppercase tracking-[0.18em] text-white/55">{label}</p>
        <div>
          <div
            className="font-display text-4xl font-semibold tracking-tight md:text-5xl"
            style={accent ? { color: accent } : undefined}
          >
            {value}
          </div>
          {sub && <div className="mt-2 text-sm text-white/60">{sub}</div>}
        </div>
      </div>
    </div>
  );
}
