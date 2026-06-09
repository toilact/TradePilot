// Mock data để dựng/giao diện chạy độc lập trước khi backend có dữ liệu thật.
import type {
  AccuracySummary,
  NewsItem,
  PricePoint,
  Prediction,
} from "./types";

export const MOCK_PREDICTIONS: Prediction[] = [
  { symbol: "VCB", name: "Vietcombank", exchange: "HOSE", sector: "Ngân hàng", close: 92.4, changePct: 1.8, label: "tang", confidence: 0.78, sentiment: 0.42, modelVersion: "v0.1.0" },
  { symbol: "FPT", name: "FPT Corp", exchange: "HOSE", sector: "Công nghệ", close: 138.2, changePct: 2.4, label: "tang", confidence: 0.81, sentiment: 0.55, modelVersion: "v0.1.0" },
  { symbol: "HPG", name: "Hòa Phát", exchange: "HOSE", sector: "Thép", close: 27.85, changePct: -0.3, label: "di_ngang", confidence: 0.64, sentiment: 0.05, modelVersion: "v0.1.0" },
  { symbol: "VHM", name: "Vinhomes", exchange: "HOSE", sector: "Bất động sản", close: 41.1, changePct: -1.6, label: "giam", confidence: 0.71, sentiment: -0.33, modelVersion: "v0.1.0" },
  { symbol: "MWG", name: "Thế Giới Di Động", exchange: "HOSE", sector: "Bán lẻ", close: 61.9, changePct: 0.7, label: "tang", confidence: 0.69, sentiment: 0.28, modelVersion: "v0.1.0" },
  { symbol: "TCB", name: "Techcombank", exchange: "HOSE", sector: "Ngân hàng", close: 24.3, changePct: 0.2, label: "di_ngang", confidence: 0.6, sentiment: 0.08, modelVersion: "v0.1.0" },
  { symbol: "SSI", name: "Chứng khoán SSI", exchange: "HOSE", sector: "Chứng khoán", close: 33.5, changePct: 3.1, label: "tang", confidence: 0.74, sentiment: 0.61, modelVersion: "v0.1.0" },
  { symbol: "GAS", name: "PV Gas", exchange: "HOSE", sector: "Năng lượng", close: 68.0, changePct: -2.1, label: "giam", confidence: 0.66, sentiment: -0.21, modelVersion: "v0.1.0" },
  { symbol: "ACB", name: "Ngân hàng ACB", exchange: "HOSE", sector: "Ngân hàng", close: 25.1, changePct: 0.4, label: "di_ngang", confidence: 0.58, sentiment: 0.02, modelVersion: "v0.1.0" },
  { symbol: "VNM", name: "Vinamilk", exchange: "HOSE", sector: "Tiêu dùng", close: 66.2, changePct: 1.1, label: "tang", confidence: 0.63, sentiment: 0.18, modelVersion: "v0.1.0" },
];

export function mockPriceSeries(symbol: string): PricePoint[] {
  // sinh chuỗi giả định ổn định theo symbol để không nhảy mỗi render
  const seed = symbol.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const out: PricePoint[] = [];
  let price = 50 + (seed % 80);
  for (let i = 60; i >= 0; i--) {
    const wobble = Math.sin((seed + i) / 6) * 1.5 + ((seed * i) % 5) - 2;
    price = Math.max(8, price + wobble * 0.4);
    const d = new Date();
    d.setDate(d.getDate() - i);
    out.push({
      date: d.toISOString().slice(0, 10),
      close: Math.round(price * 100) / 100,
      sentiment: Math.round((Math.sin((seed + i) / 9) * 0.6) * 100) / 100,
    });
  }
  return out;
}

export const MOCK_NEWS: NewsItem[] = [
  { title: "Ngân hàng đẩy mạnh tín dụng cuối quý, kỳ vọng lợi nhuận tích cực", source: "CafeF", publishedAt: "2026-06-09", sentiment: 0.5, url: "#" },
  { title: "Khối ngoại mua ròng phiên thứ ba liên tiếp", source: "FireAnt", publishedAt: "2026-06-09", sentiment: 0.35, url: "#" },
  { title: "Áp lực tỷ giá hạ nhiệt, dòng tiền quay lại nhóm vốn hóa lớn", source: "CafeF", publishedAt: "2026-06-08", sentiment: 0.2, url: "#" },
  { title: "Một số mã bất động sản chịu áp lực chốt lời ngắn hạn", source: "FireAnt", publishedAt: "2026-06-08", sentiment: -0.3, url: "#" },
];

export const MOCK_ACCURACY: AccuracySummary = {
  overall: 0.612,
  last30: 0.643,
  byLabel: { tang: 0.66, giam: 0.58, di_ngang: 0.55 },
  modelVersion: "v0.1.0",
  series: Array.from({ length: 30 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (29 - i));
    return {
      date: d.toISOString().slice(0, 10),
      accuracy: Math.round((0.55 + Math.sin(i / 4) * 0.07 + i * 0.001) * 1000) / 1000,
    };
  }),
};
