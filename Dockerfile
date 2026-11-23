# Multi-runtime production image for Nemofy Agent
# Includes:
# - Official Firecrawl Local binary built from Rust/crates.io (https://github.com/firecrawl/firecrawl#firecrawl-local)
# - Chromium for Firecrawl Local JavaScript rendering
# - Node.js 22 Gateway + React UI
# - Python 3 Agent-Core

# Stage 1: Build official `firecrawl-local` binary (Rust + C compiler + cmake per github.com/firecrawl/firecrawl#firecrawl-local)
FROM rust:1-slim-bookworm AS firecrawl-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install official firecrawl-local from crates.io (fallback to github.com/firecrawl/firecrawl apps/local)
RUN cargo install firecrawl-local --locked || ( \
    git clone --depth 1 https://github.com/firecrawl/firecrawl.git /tmp/firecrawl \
    && cd /tmp/firecrawl/apps/local \
    && cargo build --release \
    && cp target/release/firecrawl-local /usr/local/cargo/bin/firecrawl-local \
    && rm -rf /tmp/firecrawl \
    )

# Stage 2: Production runtime image
FROM node:22-bookworm-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=3000 \
    AGENT_API_PORT=8100 \
    FIRECRAWL_PORT=3002 \
    FIRECRAWL_CLOUD=auto \
    FIRECRAWL_LOCAL_BROWSER=auto \
    FIRECRAWL_CHROME_PATH=/usr/bin/chromium

# Install Python 3 runtime + Chromium (for Firecrawl Local JS rendering) + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    chromium \
    fonts-liberation \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled official `firecrawl-local` binary and verify both firecrawl-local & Chromium
COPY --from=firecrawl-builder /usr/local/cargo/bin/firecrawl-local /usr/local/bin/firecrawl-local
RUN chmod +x /usr/local/bin/firecrawl-local && firecrawl-local --version && chromium --version

WORKDIR /app

# Install Node dependencies first for layer caching
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Install Python Agent-Core dependencies into an isolated venv
COPY pyproject.toml ./
COPY Agent-Core ./Agent-Core
COPY MCPs ./MCPs
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -e .

# Copy remaining source files and build the React frontend
COPY . .
RUN npm run build

# Ensure runtime directories exist with safe permissions
RUN mkdir -p /app/data /app/data/firecrawl-local /app/workspace /app/Skills

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:3000/api/health >/dev/null || exit 1

CMD ["npx", "tsx", "server.ts"]
