// Client gọi backend FastAPI. Hiện trả mock; khi backend Phase 1.4 sẵn sàng,
// đổi USE_MOCK=false (hoặc bỏ fallback) để dùng dữ liệu thật.
// Backend là nguồn sự thật — frontend KHÔNG tự tính nhãn/sentiment.
import {
  MOCK_ACCURACY,
  MOCK_NEWS,
  MOCK_PREDICTIONS,
  mockPriceSeries,
} from "./mock";
import type { AccuracySummary, NewsItem, Prediction, PricePoint } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
// Phase 1.4 + M6: mọi hàm dùng API thật (getNews hết mock — endpoint /api/news).
// M7: revalidate 3600 — data đổi 1 lần/ngày (pipeline 16:05), ISR che cold-start Render free.
// CI build không có backend → NEXT_PUBLIC_CI_BUILD=1 buộc dùng mock để `next build` xanh.
const USE_MOCK = process.env.NEXT_PUBLIC_CI_BUILD === "1";

export async function getPredictions(): Promise<Prediction[]> {
  if (USE_MOCK) return MOCK_PREDICTIONS;
  const res = await fetch(`${API_URL}/api/predictions`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Không tải được dự đoán");
  return res.json();
}

export async function getPrediction(symbol: string): Promise<Prediction | undefined> {
  if (USE_MOCK) return MOCK_PREDICTIONS.find((p) => p.symbol === symbol.toUpperCase());
  const res = await fetch(`${API_URL}/api/predictions?symbol=${symbol}`, {
    next: { revalidate: 3600 },
  });
  if (res.status === 404) return undefined; // mã không tồn tại → trang gọi notFound()
  if (!res.ok) throw new Error("Không tải được dự đoán");
  return res.json();
}

export async function getHistory(symbol: string): Promise<PricePoint[]> {
  if (USE_MOCK) return mockPriceSeries(symbol);
  const res = await fetch(`${API_URL}/api/stocks/${symbol}/history`, {
    next: { revalidate: 3600 },
  });
  if (!res.ok) throw new Error("Không tải được lịch sử");
  return res.json();
}

export async function getNews(symbol: string): Promise<NewsItem[]> {
  if (USE_MOCK) return MOCK_NEWS;
  const res = await fetch(`${API_URL}/api/news?symbol=${symbol}&limit=10`, {
    next: { revalidate: 3600 },
  });
  if (res.status === 404) return []; // mã không tồn tại → trang đã notFound() qua getPrediction
  if (!res.ok) throw new Error("Không tải được tin tức");
  return res.json();
}

export async function getAccuracy(): Promise<AccuracySummary> {
  if (USE_MOCK) return MOCK_ACCURACY;
  const res = await fetch(`${API_URL}/api/accuracy`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Không tải được độ chính xác");
  return res.json();
}
