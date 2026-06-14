"use client";

// Nguồn trạng thái watchlist dùng chung (M9) — nạp 1 lần khi đăng nhập (tránh 30 nút × 1 fetch),
// chia sẻ tập symbol cho mọi WatchlistButton + cập nhật lạc quan (optimistic) khi toggle.
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { addToWatchlist, getWatchlist, removeFromWatchlist } from "@/lib/api";

interface WatchlistCtx {
  symbols: Set<string>;
  ready: boolean; // đã biết trạng thái (đã fetch xong hoặc chắc chắn chưa đăng nhập)
  has: (symbol: string) => boolean;
  toggle: (symbol: string) => Promise<void>;
}

const Ctx = createContext<WatchlistCtx | null>(null);

export function WatchlistProvider({ children }: { children: React.ReactNode }) {
  const { status } = useSession();
  const [symbols, setSymbols] = useState<Set<string>>(new Set());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    if (status === "authenticated") {
      setReady(false);
      getWatchlist()
        .then((items) => {
          if (active) setSymbols(new Set(items.map((i) => i.symbol.toUpperCase())));
        })
        .finally(() => active && setReady(true));
    } else {
      setSymbols(new Set());
      setReady(status === "unauthenticated");
    }
    return () => {
      active = false;
    };
  }, [status]);

  const toggle = useCallback(async (symbol: string) => {
    const s = symbol.toUpperCase();
    let wasIn = false;
    setSymbols((prev) => {
      wasIn = prev.has(s);
      const next = new Set(prev);
      if (wasIn) next.delete(s);
      else next.add(s);
      return next;
    });
    try {
      if (wasIn) await removeFromWatchlist(s);
      else await addToWatchlist(s);
    } catch {
      // rollback nếu API lỗi
      setSymbols((prev) => {
        const next = new Set(prev);
        if (wasIn) next.add(s);
        else next.delete(s);
        return next;
      });
    }
  }, []);

  return (
    <Ctx.Provider value={{ symbols, ready, has: (s) => symbols.has(s.toUpperCase()), toggle }}>
      {children}
    </Ctx.Provider>
  );
}

export function useWatchlist(): WatchlistCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useWatchlist phải nằm trong <WatchlistProvider>");
  return ctx;
}
