"use client";

// Watchlist thật (M9): gate đăng nhập (Auth.js Google) → fetch mã user theo dõi → xoá được.
// Trạng thái ☆/★ chia sẻ qua WatchlistProvider; danh sách (kèm tên) fetch riêng để hiển thị.
import { signIn, useSession } from "next-auth/react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Reveal } from "@/components/reveal";
import { useWatchlist } from "@/components/watchlist-provider";
import { getWatchlist } from "@/lib/api";
import type { WatchlistItem } from "@/lib/types";

export default function WatchlistPage() {
  const { status } = useSession();
  const { symbols, toggle } = useWatchlist();
  const [items, setItems] = useState<WatchlistItem[] | null>(null);

  useEffect(() => {
    if (status !== "authenticated") {
      setItems(null);
      return;
    }
    let active = true;
    getWatchlist()
      .then((data) => active && setItems(data))
      .catch(() => active && setItems([]));
    return () => {
      active = false;
    };
  }, [status]);

  return (
    <div className="mx-auto max-w-3xl px-4 pt-32 md:px-8">
      <Reveal>
        <span className="eyebrow">Cá nhân hóa</span>
      </Reveal>
      <Reveal delay={80}>
        <h1 className="mt-6 font-display text-5xl font-semibold tracking-tight text-white">
          Danh sách theo dõi
        </h1>
      </Reveal>

      <Reveal delay={140}>
        <div className="mt-12 bezel-shell">
          <div className="bezel-core px-6 py-8 md:px-8">
            {status === "loading" || (status === "authenticated" && items === null) ? (
              <Loading />
            ) : status !== "authenticated" ? (
              <SignedOut />
            ) : (
              <SignedIn items={items!.filter((i) => symbols.has(i.symbol))} onRemove={toggle} />
            )}
          </div>
        </div>
      </Reveal>
    </div>
  );
}

function Loading() {
  return (
    <div className="flex flex-col items-center gap-4 py-16 text-center text-white/50">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/15 border-t-white/60" />
      <p className="text-sm">Đang tải danh sách…</p>
    </div>
  );
}

function SignedOut() {
  return (
    <div className="flex flex-col items-center gap-6 py-14 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-2xl text-gold">
        ☆
      </div>
      <div>
        <h2 className="font-display text-xl font-semibold text-white">
          Đăng nhập để theo dõi mã của bạn
        </h2>
        <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-white/60">
          Lưu các mã quan tâm và xem nhanh dự đoán T+1 của chúng. Đăng nhập bằng Google.
        </p>
      </div>
      <button
        type="button"
        onClick={() => signIn("google")}
        className="group inline-flex items-center gap-2 rounded-full bg-white py-2.5 pl-5 pr-2 text-sm font-medium text-black transition-all duration-500 ease-fluid active:scale-[0.98]"
      >
        Đăng nhập với Google
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-black/10 transition-transform duration-500 ease-spring group-hover:translate-x-1 group-hover:-translate-y-[1px]">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M7 17L17 7M17 7H9M17 7V15" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
    </div>
  );
}

function SignedIn({
  items,
  onRemove,
}: {
  items: WatchlistItem[];
  onRemove: (symbol: string) => void;
}) {
  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center gap-4 py-14 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-xl text-white/40">
          ☆
        </div>
        <p className="max-w-sm text-sm leading-relaxed text-white/60">
          Chưa có mã nào. Bấm ngôi sao ☆ ở bảng dự đoán hoặc trang chi tiết để thêm vào đây.
        </p>
        <Link
          href="/"
          className="rounded-full border border-white/10 px-5 py-2.5 text-sm text-white/75 transition-colors hover:bg-white/5 hover:text-white"
        >
          Khám phá dự đoán
        </Link>
      </div>
    );
  }

  return (
    <ul className="divide-y divide-white/5">
      {items.map((it) => (
        <li key={it.symbol} className="flex items-center justify-between gap-4 py-4">
          <Link href={`/stock/${it.symbol}`} className="group flex items-center gap-3 min-w-0">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] font-display text-xs font-semibold text-white/80">
              {it.symbol.slice(0, 3)}
            </span>
            <span className="min-w-0">
              <span className="flex items-center gap-2">
                <span className="font-display text-sm font-semibold text-white">{it.symbol}</span>
                <span className="text-[10px] uppercase tracking-wider text-white/45">
                  {it.exchange}
                </span>
              </span>
              <span className="block truncate text-xs text-white/60">{it.name}</span>
            </span>
          </Link>
          <button
            type="button"
            onClick={() => onRemove(it.symbol)}
            className="shrink-0 rounded-full border border-white/10 px-3 py-1.5 text-xs text-white/55 transition-colors hover:border-down/40 hover:text-down"
          >
            Bỏ theo dõi
          </button>
        </li>
      ))}
    </ul>
  );
}
