// Mesh gradient nền — orb tĩnh blur, drift nhẹ. Fixed + z thấp, không repaint khi scroll.
export function MeshBackground() {
  return (
    <div className="mesh-bg" aria-hidden>
      {/* 2 quầng champagne/gold rất mờ — thống nhất với màu thương hiệu */}
      <div
        className="mesh-orb animate-orb-drift"
        style={{
          top: "-12%",
          left: "-6%",
          width: "48vw",
          height: "48vw",
          opacity: 0.28,
          background: "radial-gradient(circle, rgba(232,195,158,0.55), transparent 66%)",
        }}
      />
      <div
        className="mesh-orb animate-orb-drift"
        style={{
          bottom: "-18%",
          right: "-10%",
          width: "52vw",
          height: "52vw",
          opacity: 0.22,
          background: "radial-gradient(circle, rgba(212,175,135,0.5), transparent 66%)",
          animationDelay: "-9s",
        }}
      />
      {/* Vignette để chữ luôn rõ — khớp nền #0F1115 */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 80% at 50% 0%, transparent 42%, rgba(15,17,21,0.75) 100%)",
        }}
      />
    </div>
  );
}
