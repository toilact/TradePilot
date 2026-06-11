// Sentry phía browser — chỉ init khi có DSN (dev/CI không cần, build vẫn xanh).
import * as Sentry from "@sentry/nextjs";

if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
  Sentry.init({
    dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
    environment: process.env.NODE_ENV,
    tracesSampleRate: 0, // chỉ bắt lỗi, không APM — đúng phạm vi M4, tiết kiệm quota free
    sendDefaultPii: false,
  });
}

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
