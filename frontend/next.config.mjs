import { withSentryConfig } from "@sentry/nextjs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// Không có SENTRY_AUTH_TOKEN → bỏ qua upload source map (build local/CI vẫn xanh);
// runtime init vẫn gate theo NEXT_PUBLIC_SENTRY_DSN trong instrumentation-client.ts.
export default withSentryConfig(nextConfig, {
  silent: true,
  webpack: { treeshake: { removeDebugLogging: true } },
  widenClientFileUpload: false,
  telemetry: false,
});
