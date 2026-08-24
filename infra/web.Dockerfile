# syntax=docker/dockerfile:1
# SentenAI — Web image (Next.js dashboard)

FROM node:22-slim AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci

FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web ./
# Requires next.config.ts's `output: "standalone"` — trims the final image to just the
# files actually reachable at runtime instead of the whole node_modules tree.
RUN npm run build

# ---- runtime: standalone server only, no npm/devDependencies, non-root ----
FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN groupadd --system nextjs && \
    useradd --system --gid nextjs --home-dir /app --shell /usr/sbin/nologin nextjs

COPY --from=build /app/public ./public
COPY --from=build --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nextjs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/login',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
