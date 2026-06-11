"use client";

// Lưới an toàn cuối cùng của App Router: lỗi render ở root layout rơi vào đây.
// Phải tự render <html>/<body> vì thay thế toàn bộ layout. Báo Sentry rồi cho reset.
import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="vi">
      <body style={{ background: "#0F1115", color: "#E8EAED", fontFamily: "sans-serif" }}>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            textAlign: "center",
            padding: 24,
          }}
        >
          <h1 style={{ fontSize: 28, fontWeight: 600 }}>Có lỗi xảy ra</h1>
          <p style={{ color: "rgba(255,255,255,0.5)" }}>
            Sự cố đã được ghi nhận. Vui lòng thử lại.
          </p>
          <button
            onClick={() => reset()}
            style={{
              background: "#fff",
              color: "#000",
              border: 0,
              borderRadius: 999,
              padding: "10px 20px",
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Thử lại
          </button>
        </div>
      </body>
    </html>
  );
}
