// Khớp với backend (PLAN.md). Nhãn ±1%: tang / giam / di_ngang.
export type Label = "tang" | "giam" | "di_ngang";

// Trạng thái hiển thị (M3 — confidence-gating): backend quyết định, frontend chỉ render.
// "khong_du_tin_hieu" = model không đủ tự tin (confidence < threshold).
export type Display = Label | "khong_du_tin_hieu";

export type Exchange = "HOSE" | "HNX" | "UPCOM";

export interface Prediction {
  symbol: string;
  name: string;
  exchange: Exchange;
  sector: string;
  close: number; // giá đóng cửa gần nhất (nghìn đồng)
  changePct: number; // % thay đổi phiên gần nhất
  label: Label; // dự đoán T+1 (argmax — luôn trả để minh bạch prob)
  confidence: number; // 0..1, max prob SAU calibration
  sentiment: number; // -1..1, sentiment tổng hợp ngày gần nhất
  modelVersion: string;
  // --- Confidence-gating (M3) — null với bản ghi stub_v0 cũ ---
  probTang: number | null;
  probGiam: number | null;
  probDiNgang: number | null;
  isActionable: boolean;
  threshold: number | null;
  display: Display; // badge render theo field này, KHÔNG tự suy từ threshold
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

// M9 — mã trong watchlist của user (backend trả sau khi verify JWT).
export interface WatchlistItem {
  symbol: string;
  name: string;
  exchange: string;
}

export interface AccuracyPoint {
  date: string;
  accuracy: number; // 0..1, rolling
}

// M10 — accuracy trượt 30 phiên theo từng model version (giám sát drift theo version).
export interface VersionRolling {
  version: string;
  points: AccuracyPoint[];
}

export interface AccuracySummary {
  overall: number;
  last30: number;
  byLabel: Record<Label, number>;
  series: AccuracyPoint[];
  rollingByVersion: VersionRolling[]; // M10 — multi-line chart theo version
  modelVersion: string;
  coverage: number; // % dự đoán dám đoán (is_actionable) của version hiện hành
  precisionActionable: number | null; // accuracy trên tập dám đoán (null = chưa có actual)
}
