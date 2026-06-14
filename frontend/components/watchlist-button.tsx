"use client";

// Nút ☆/★ theo dõi 1 mã (M9) — dùng ở trang chủ (overlay trên hàng) + trang chi tiết.
// Chưa đăng nhập → prompt signIn Google. Đã đăng nhập → toggle qua WatchlistProvider.
import { signIn, useSession } from "next-auth/react";
import { useWatchlist } from "./watchlist-provider";

export function WatchlistButton({
  symbol,
  withLabel = false,
  className = "",
}: {
  symbol: string;
  withLabel?: boolean;
  className?: string;
}) {
  const { status } = useSession();
  const { has, toggle } = useWatchlist();
  const active = status === "authenticated" && has(symbol);

  function onClick(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (status !== "authenticated") {
      signIn("google");
      return;
    }
    void toggle(symbol);
  }

  const label = active ? "Đang theo dõi" : "Theo dõi";

  if (withLabel) {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-pressed={active}
        className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition-colors duration-300 ease-fluid ${
          active
            ? "border-gold/40 bg-gold/10 text-gold"
            : "border-white/10 text-white/70 hover:bg-white/5 hover:text-white"
        } ${className}`}
      >
        <span aria-hidden className="text-base leading-none">
          {active ? "★" : "☆"}
        </span>
        {label}
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={`flex h-8 w-8 items-center justify-center rounded-full text-base leading-none transition-colors duration-300 ease-fluid ${
        active ? "text-gold" : "text-white/35 hover:text-white/80"
      } ${className}`}
    >
      <span aria-hidden>{active ? "★" : "☆"}</span>
    </button>
  );
}
