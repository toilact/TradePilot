"use client";

// Nút đăng nhập/đăng xuất Google (M9). Hiện trong nav; ẩn token, chỉ hiện email.
import { signIn, signOut, useSession } from "next-auth/react";

export function AuthButton() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return <span className="px-3 text-sm text-white/35">…</span>;
  }

  if (session?.user) {
    return (
      <div className="flex items-center gap-2">
        <span className="hidden max-w-[10rem] truncate text-xs text-white/55 lg:inline">
          {session.user.email}
        </span>
        <button
          type="button"
          onClick={() => signOut()}
          className="rounded-full border border-white/10 px-4 py-2 text-sm text-white/70 transition-colors duration-300 ease-fluid hover:bg-white/5 hover:text-white"
        >
          Đăng xuất
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => signIn("google")}
      className="rounded-full border border-white/10 px-4 py-2 text-sm text-white/70 transition-colors duration-300 ease-fluid hover:bg-white/5 hover:text-white"
    >
      Đăng nhập
    </button>
  );
}
