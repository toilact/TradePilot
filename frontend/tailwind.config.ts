import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-jakarta)", "system-ui", "sans-serif"],
        display: ["var(--font-space)", "system-ui", "sans-serif"],
      },
      colors: {
        // Semantic nhãn dự đoán
        up: "#34d399", // emerald-400 — Tăng
        down: "#fb7185", // rose-400 — Giảm
        flat: "#a1a1aa", // zinc-400 — Đi ngang
        gold: "#e8c39e", // accent sang trọng
      },
      transitionTimingFunction: {
        fluid: "cubic-bezier(0.32, 0.72, 0, 1)",
        spring: "cubic-bezier(0.16, 1, 0.3, 1)",
      },
      borderRadius: {
        squircle: "2rem",
      },
      keyframes: {
        "orb-drift": {
          "0%, 100%": { transform: "translate3d(0,0,0) scale(1)" },
          "50%": { transform: "translate3d(4%, -6%, 0) scale(1.08)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "orb-drift": "orb-drift 18s ease-in-out infinite",
        shimmer: "shimmer 2.5s infinite",
      },
    },
  },
  plugins: [],
};

export default config;
