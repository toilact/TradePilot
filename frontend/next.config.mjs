import { withSentryConfig } from "@sentry/nextjs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// Chỉ áp dụng withSentryConfig khi có SENTRY_AUTH_TOKEN (tránh lỗi build Vercel khi chưa cấu hình token Sentry)
export default process.env.SENTRY_AUTH_TOKEN
  ? withSentryConfig(nextConfig, {
      silent: true,
      webpack: { treeshake: { removeDebugLogging: true } },
      widenClientFileUpload: false,
      telemetry: false,
    })
  : nextConfig;

