"use client";

import Link from "next/link";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Dự đoán" },
  { href: "/accuracy", label: "Độ chính xác" },
  { href: "/watchlist", label: "Theo dõi" },
];

export function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header className="pointer-events-none fixed inset-x-0 top-0 z-40 flex justify-center">
      {/* Fluid island pill — tách khỏi mép trên */}
      <nav className="pointer-events-auto mx-auto mt-6 flex w-max items-center gap-1 rounded-full border border-white/10 bg-black/40 px-2 py-2 backdrop-blur-2xl">
        <Link
          href="/"
          className="flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium tracking-tight"
        >
          <span className="font-display text-base font-semibold text-white">
            Trade<span className="text-gold">Pilot</span>
          </span>
        </Link>

        <div className="mx-1 hidden h-5 w-px bg-white/10 md:block" />

        <div className="hidden items-center gap-0.5 md:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="rounded-full px-4 py-2 text-sm text-white/60 transition-colors duration-300 ease-fluid hover:bg-white/5 hover:text-white"
            >
              {l.label}
            </Link>
          ))}
        </div>

        {/* CTA island button-in-button */}
        <Link
          href="/"
          className="group ml-1 hidden items-center gap-2 rounded-full bg-white py-2 pl-5 pr-2 text-sm font-medium text-black transition-all duration-500 ease-fluid active:scale-[0.98] md:flex"
        >
          Xem hôm nay
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-black/10 transition-transform duration-500 ease-spring group-hover:translate-x-1 group-hover:-translate-y-[1px]">
            <Arrow />
          </span>
        </Link>

        {/* Hamburger (mobile) */}
        <button
          onClick={() => setOpen((v) => !v)}
          aria-label="Mở menu"
          className="relative flex h-10 w-10 items-center justify-center rounded-full transition-colors hover:bg-white/5 md:hidden"
        >
          <span
            className={`absolute h-px w-5 bg-white transition-all duration-500 ease-fluid ${
              open ? "rotate-45" : "-translate-y-1.5"
            }`}
          />
          <span
            className={`absolute h-px w-5 bg-white transition-all duration-500 ease-fluid ${
              open ? "-rotate-45" : "translate-y-1.5"
            }`}
          />
        </button>
      </nav>

      {/* Modal expansion overlay */}
      <div
        className={`pointer-events-auto fixed inset-0 z-30 flex flex-col items-center justify-center gap-2 bg-black/80 backdrop-blur-3xl transition-all duration-500 ease-fluid md:hidden ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        {LINKS.map((l, i) => (
          <Link
            key={l.href}
            href={l.href}
            onClick={() => setOpen(false)}
            className="font-display text-3xl font-medium text-white/90 transition-all duration-500 ease-fluid"
            style={{
              transitionDelay: open ? `${100 + i * 60}ms` : "0ms",
              transform: open ? "translateY(0)" : "translateY(3rem)",
              opacity: open ? 1 : 0,
            }}
          >
            {l.label}
          </Link>
        ))}
      </div>
    </header>
  );
}

function Arrow() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M7 17L17 7M17 7H9M17 7V15" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
