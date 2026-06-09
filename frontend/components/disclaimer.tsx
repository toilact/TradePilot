// Bất biến dự án: mọi nơi hiển thị dự đoán PHẢI có disclaimer này.
export function Disclaimer({ variant = "inline" }: { variant?: "inline" | "banner" }) {
  if (variant === "banner") {
    return (
      <div className="bezel-shell">
        <div className="bezel-core flex items-start gap-3 px-5 py-4">
          <span className="mt-0.5 text-gold/80" aria-hidden>
            <InfoIcon />
          </span>
          <p className="text-sm leading-relaxed text-white/65">
            <span className="font-medium text-white/75">Đây không phải khuyến nghị đầu tư.</span>{" "}
            Dự đoán do mô hình thống kê sinh ra từ dữ liệu lịch sử và sentiment tin tức, có thể sai.
            Bạn tự chịu trách nhiệm cho mọi quyết định giao dịch.
          </p>
        </div>
      </div>
    );
  }
  return (
    <p className="text-xs leading-relaxed text-white/55">
      ⚠ Đây không phải khuyến nghị đầu tư — dự đoán có thể sai, bạn tự chịu trách nhiệm.
    </p>
  );
}

function InfoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" strokeLinecap="round" />
    </svg>
  );
}
