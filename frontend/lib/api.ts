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
const USE_MOCK = false; // Phase 1.4: dùng API thật. getNews vẫn fallback mock (chưa có endpoint tin).

export async function getPredictions(): Promise<Prediction[]> {
  if (USE_MOCK) return MOCK_PREDICTIONS;
  const res = await fetch(`${API_URL}/api/predictions`, { next: { revalidate: 300 } });
  if (!res.ok) throw new Error("Không tải được dự đoán");
  return res.json();
}

export async function getPrediction(symbol: string): Promise<Prediction | undefined> {
  if (USE_MOCK) return MOCK_PREDICTIONS.find((p) => p.symbol === symbol.toUpperCase());
  const res = await fetch(`${API_URL}/api/predictions?symbol=${symbol}`, {
    next: { revalidate: 300 },
  });
  if (res.status === 404) return undefined; // mã không tồn tại → trang gọi notFound()
  if (!res.ok) throw new Error("Không tải được dự đoán");
  return res.json();
}

export async function getHistory(symbol: string): Promise<PricePoint[]> {
  if (USE_MOCK) return mockPriceSeries(symbol);
  const res = await fetch(`${API_URL}/api/stocks/${symbol}/history`, {
    next: { revalidate: 300 },
  });
  if (!res.ok) throw new Error("Không tải được lịch sử");
  return res.json();
}

export async function getNews(_symbol: string): Promise<NewsItem[]> {
  if (USE_MOCK) return MOCK_NEWS;
  return MOCK_NEWS; // TODO: endpoint tin tức theo mã
}

export async function getAccuracy(): Promise<AccuracySummary> {
  if (USE_MOCK) return MOCK_ACCURACY;
  const res = await fetch(`${API_URL}/api/accuracy`, { next: { revalidate: 600 } });
  if (!res.ok) throw new Error("Không tải được độ chính xác");
  return res.json();
}
