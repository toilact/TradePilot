import Link from "next/link";
import { Disclaimer } from "./disclaimer";

export function SiteFooter() {
  return (
    <footer className="relative mx-auto mt-32 max-w-6xl px-4 pb-16 md:px-8">
      <div className="bezel-shell">
        <div className="bezel-core grid gap-10 px-6 py-10 md:grid-cols-[1.5fr_1fr_1fr] md:px-10">
          <div>
            <span className="font-display text-lg font-semibold text-white">
              Trade<span className="text-gold">Pilot</span>
            </span>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-white/60">
              Dự đoán xu hướng cổ phiếu Việt Nam T+1 từ giá lịch sử và sentiment tin tức tiếng Việt.
            </p>
          </div>
          <FooterCol
            title="Sản phẩm"
            links={[
              { href: "/", label: "Dự đoán hôm nay" },
              { href: "/accuracy", label: "Độ chính xác" },
              { href: "/watchlist", label: "Danh sách theo dõi" },
            ]}
          />
          <FooterCol
            title="Dự án"
            links={[
              { href: "/", label: "Giới thiệu" },
              { href: "/", label: "Phương pháp" },
            ]}
          />
        </div>
      </div>

      <div className="mt-6 px-2">
        <Disclaimer variant="inline" />
        <p className="mt-3 text-xs text-white/45">
          © {new Date().getFullYear()} TradePilot · Dự án học tập cá nhân.
        </p>
      </div>
    </footer>
  );
}

function FooterCol({
  title,
  links,
}: {
  title: string;
  links: { href: string; label: string }[];
}) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-[0.18em] text-white/50">{title}</p>
      <ul className="mt-4 space-y-2.5">
        {links.map((l, i) => (
          <li key={`${l.href}-${i}`}>
            <Link
              href={l.href}
              className="text-sm text-white/65 transition-colors duration-300 ease-fluid hover:text-white"
            >
              {l.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
