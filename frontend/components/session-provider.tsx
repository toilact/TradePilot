"use client";

// Bọc SessionProvider của Auth.js để useSession() dùng được ở client components (M9).
import { SessionProvider } from "next-auth/react";

export function AuthSessionProvider({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}
