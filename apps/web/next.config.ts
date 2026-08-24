import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle (only the deps actually reachable at runtime) — lets
  // infra/web.Dockerfile ship a minimal image instead of the full node_modules tree.
  output: "standalone",
};

export default nextConfig;
