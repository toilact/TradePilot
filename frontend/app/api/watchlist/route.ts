// BFF proxy watchlist (M9) — browser gọi same-origin route này; ta đọc cookie session HS256
// (httpOnly) server-side rồi forward `Authorization: Bearer` sang FastAPI. Token KHÔNG lộ ra
// JS phía client, và browser không phải gọi cross-origin Render (tránh CORS/credentials).
import { cookies } from "next/headers";
import { NextResponse, type NextRequest } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const dynamic = "force-dynamic"; // per-user, không bao giờ cache/prerender

async function sessionToken(): Promise<string | null> {
  const jar = await cookies();
  // prod (https) = __Secure-authjs.session-token; dev = authjs.session-token.
  return (
    jar.get("__Secure-authjs.session-token")?.value ??
    jar.get("authjs.session-token")?.value ??
    null
  );
}

async function forward(method: string, search = "", body?: string) {
  const token = await sessionToken();
  if (!token) {
    return NextResponse.json({ detail: "Chưa đăng nhập" }, { status: 401 });
  }
  const res = await fetch(`${API_URL}/api/watchlist${search}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body,
    cache: "no-store",
  });
  const data = await res.json().catch(() => ({}));
  return NextResponse.json(data, { status: res.status });
}

export async function GET() {
  return forward("GET");
}

export async function POST(req: NextRequest) {
  return forward("POST", "", await req.text());
}

export async function DELETE(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol") ?? "";
  return forward("DELETE", `?symbol=${encodeURIComponent(symbol)}`);
}
