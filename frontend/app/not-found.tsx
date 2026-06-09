import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-[80vh] max-w-2xl flex-col items-center justify-center px-4 text-center">
      <span className="eyebrow">404</span>
      <h1 className="mt-6 font-display text-5xl font-semibold tracking-tight text-white">
        Không tìm thấy mã này
      </h1>
      <p className="mt-4 text-white/50">Mã cổ phiếu không tồn tại hoặc chưa được theo dõi.</p>
      <Link
        href="/"
        className="group mt-8 inline-flex items-center gap-2 rounded-full bg-white py-2.5 pl-5 pr-2 text-sm font-medium text-black transition-all duration-500 ease-fluid active:scale-[0.98]"
      >
        Về trang dự đoán
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-black/10 transition-transform duration-500 ease-spring group-hover:translate-x-1">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M7 17L17 7M17 7H9M17 7V15" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </Link>
    </div>
  );
}
