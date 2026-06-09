import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";
import { SiteFooter } from "@/components/site-footer";
import { MeshBackground } from "@/components/mesh-background";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin", "vietnamese"],
  variable: "--font-jakarta",
  display: "swap",
});

const space = Space_Grotesk({
  subsets: ["latin", "vietnamese"],
  variable: "--font-space",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TradePilot — Dự đoán cổ phiếu Việt Nam",
  description:
    "Dự đoán xu hướng T+1 (Tăng / Giảm / Đi ngang) cho cổ phiếu Việt Nam từ giá lịch sử và sentiment tin tức. Đây không phải khuyến nghị đầu tư.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" className={`${jakarta.variable} ${space.variable}`}>
      <body className="min-h-[100dvh] font-sans antialiased">
        <MeshBackground />
        <div className="grain" aria-hidden />
        <Nav />
        <main className="relative">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
