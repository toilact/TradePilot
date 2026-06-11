import { getPredictions } from "@/lib/api";
import { Reveal } from "@/components/reveal";
import { StatCard } from "@/components/stat-card";
import { PredictionTable } from "@/components/prediction-table";
import { Disclaimer } from "@/components/disclaimer";
import { fmtConfidence } from "@/lib/format";

export default async function Home() {
  const preds = await getPredictions();
  const today = new Date().toLocaleDateString("vi-VN", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  // Đếm theo `display` — chỉ tín hiệu model dám đoán (gating M3); mã dưới ngưỡng
  // mang trạng thái "không đủ tín hiệu", không tính là tín hiệu Tăng/Giảm.
  const up = preds.filter((p) => p.display === "tang").length;
  const down = preds.filter((p) => p.display === "giam").length;
  const flat = preds.filter((p) => p.display === "di_ngang").length;
  const noSignal = preds.filter((p) => !p.isActionable).length;
  const avgConf = preds.reduce((a, p) => a + p.confidence, 0) / preds.length;

  return (
    <div className="mx-auto max-w-6xl px-4 md:px-8">
      {/* ---- Hero ---- */}
      <section className="flex min-h-[70vh] flex-col justify-center pt-32 pb-16">
        <Reveal>
          <span className="eyebrow">Phiên T+1 · {today}</span>
        </Reveal>
        <Reveal delay={80}>
          <h1 className="mt-6 max-w-3xl text-balance font-display text-5xl font-semibold leading-[1.05] tracking-tight text-white md:text-7xl">
            Dự đoán xu hướng cổ phiếu Việt Nam,{" "}
            <span className="text-white/65">trước khi phiên mở cửa.</span>
          </h1>
        </Reveal>
        <Reveal delay={160}>
          <p className="mt-7 max-w-xl text-lg leading-relaxed text-white/65">
            Mô hình kết hợp giá lịch sử và sentiment tin tức tiếng Việt để ước lượng khả năng
            Tăng, Giảm hay Đi ngang cho phiên kế tiếp.
          </p>
        </Reveal>
        <Reveal delay={240}>
          <div className="mt-9 max-w-xl">
            <Disclaimer variant="banner" />
          </div>
        </Reveal>
      </section>

      {/* ---- Bento thống kê (asymmetrical) ---- */}
      <section className="grid grid-cols-1 gap-4 md:grid-cols-4 md:grid-rows-2">
        <Reveal className="md:col-span-2 md:row-span-2">
          <StatCard
            label="Tín hiệu nổi bật hôm nay"
            value={<span className="text-up">{up} mã Tăng</span>}
            sub={
              <span>
                {down} mã dự đoán Giảm · {flat} mã Đi ngang · {noSignal} mã không đủ tín hiệu
                trong {preds.length} mã theo dõi.
              </span>
            }
            className="min-h-[220px]"
          />
        </Reveal>
        <Reveal delay={80} className="md:col-span-2">
          <StatCard
            label="Độ tin cậy trung bình"
            value={fmtConfidence(avgConf)}
            sub="Trên toàn bộ dự đoán phiên này"
            accent="#e8c39e"
          />
        </Reveal>
        <Reveal delay={140} className="md:col-span-1">
          <StatCard label="Tăng" value={<span className="text-up">{up}</span>} />
        </Reveal>
        <Reveal delay={200} className="md:col-span-1">
          <StatCard label="Giảm" value={<span className="text-down">{down}</span>} />
        </Reveal>
      </section>

      {/* ---- Bảng dự đoán ---- */}
      <section className="mt-24">
        <Reveal>
          <div className="mb-8 flex items-end justify-between">
            <div>
              <span className="eyebrow">Bảng dự đoán</span>
              <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight text-white md:text-4xl">
                Toàn bộ tín hiệu phiên tới
              </h2>
            </div>
            <span className="hidden text-sm text-white/65 md:block">
              Cập nhật ~16:00 sau đóng cửa
            </span>
          </div>
        </Reveal>

        <Reveal delay={100}>
          <PredictionTable preds={preds} />
        </Reveal>
      </section>
    </div>
  );
}
