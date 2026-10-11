import path from "node:path";

import type { NextConfig } from "next";

// API calls stay same-origin (/api/...) so auth cookies work without CORS tricks.
// In production Nginx routes /api straight to the backend; this rewrite covers dev and docker.
const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8010";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // Agent answers stream for up to ~2 minutes on a CPU-only local model.
    proxyTimeout: 180_000,
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    root: path.join(__dirname),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },
};

export default nextConfig;
