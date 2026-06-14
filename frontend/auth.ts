// Auth.js v5 — Google-only (M9). Session JWT ký HS256 shared-secret (KHÔNG dùng JWE mặc định)
// để FastAPI verify được bằng cùng AUTH_SECRET. Xem backend/api/auth.py::get_current_user.
//
// Hợp đồng JWT (phải khớp backend): alg HS256, key = bytes UTF-8 thô của AUTH_SECRET, claim
// `email` (bắt buộc) + `name`. BFF proxy (app/api/watchlist/route.ts) đọc cookie HS256 này
// rồi forward `Authorization: Bearer` sang FastAPI — token giữ httpOnly, không lộ ra JS.
import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import type { JWT } from "next-auth/jwt";
import { SignJWT, jwtVerify } from "jose";

const SECRET = process.env.AUTH_SECRET ?? "";
const KEY = new TextEncoder().encode(SECRET);
const MAX_AGE = 30 * 24 * 60 * 60; // 30 ngày (giây) — khớp session.maxAge

export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: true, // build/deploy không cần set AUTH_URL (Vercel/Render đứng sau proxy)
  secret: SECRET,
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  ],
  session: { strategy: "jwt", maxAge: MAX_AGE },
  jwt: {
    // Override encode/decode: cookie session = JWS HS256 (không JWE) → backend verify cùng secret.
    async encode({ token }) {
      return await new SignJWT({ ...(token as Record<string, unknown>) })
        .setProtectedHeader({ alg: "HS256" })
        .setIssuedAt()
        .setExpirationTime(`${MAX_AGE}s`)
        .sign(KEY);
    },
    async decode({ token }) {
      if (!token) return null;
      try {
        const { payload } = await jwtVerify(token, KEY, { algorithms: ["HS256"] });
        return payload as unknown as JWT;
      } catch {
        return null; // token hỏng/hết hạn → coi như chưa đăng nhập
      }
    },
  },
  callbacks: {
    // Đảm bảo email/name nằm trong token (backend lấy email làm khóa upsert users).
    async jwt({ token, profile }) {
      if (profile) {
        token.email = profile.email ?? token.email;
        token.name = profile.name ?? token.name;
      }
      return token;
    },
  },
});
