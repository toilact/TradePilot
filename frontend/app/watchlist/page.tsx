import { Reveal } from "@/components/reveal";
import Link from "next/link";

// Phase 3: yêu cầu đăng nhập (NextAuth) + lưu watchlist qua backend.
// Hiện là placeholder empty-state cao cấp.
export default function WatchlistPage() {
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
          <div className="bezel-core flex flex-col items-center gap-6 px-8 py-20 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-2xl text-gold">
              ☆
            </div>
            <div>
              <h2 className="font-display text-xl font-semibold text-white">
                Đăng nhập để theo dõi mã của bạn
              </h2>
              <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-white/60">
                Lưu các mã quan tâm và nhận tín hiệu dự đoán mạnh. Tính năng đăng nhập (Email &
                Google) sẽ có ở Phase 3.
              </p>
            </div>
            <Link
              href="/"
              className="group inline-flex items-center gap-2 rounded-full bg-white py-2.5 pl-5 pr-2 text-sm font-medium text-black transition-all duration-500 ease-fluid active:scale-[0.98]"
            >
              Khám phá dự đoán
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-black/10 transition-transform duration-500 ease-spring group-hover:translate-x-1 group-hover:-translate-y-[1px]">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M7 17L17 7M17 7H9M17 7V15" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            </Link>
          </div>
        </div>
      </Reveal>
    </div>
  );
}
