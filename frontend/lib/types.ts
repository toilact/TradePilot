// Khớp với backend (PLAN.md). Nhãn ±1%: tang / giam / di_ngang.
export type Label = "tang" | "giam" | "di_ngang";

export type Exchange = "HOSE" | "HNX" | "UPCOM";

export interface Prediction {
  symbol: string;
  name: string;
  exchange: Exchange;
  sector: string;
  close: number; // giá đóng cửa gần nhất (nghìn đồng)
  changePct: number; // % thay đổi phiên gần nhất
  label: Label; // dự đoán T+1
  confidence: number; // 0..1
  sentiment: number; // -1..1, sentiment tổng hợp ngày gần nhất
  modelVersion: string;
}

export interface PricePoint {
  date: string; // YYYY-MM-DD
  close: number;
  sentiment: number; // -1..1 (0 nếu ngày không có tin)
}

export interface NewsItem {
  title: string;
  source: string;
  publishedAt: string;
  sentiment: number;
  url: string;
}

export interface AccuracyPoint {
  date: string;
  accuracy: number; // 0..1, rolling
}

export interface AccuracySummary {
  overall: number;
  last30: number;
  byLabel: Record<Label, number>;
  series: AccuracyPoint[];
  modelVersion: string;
}
