# syntax=docker/dockerfile:1
# Multi-runtime production image for Nemofy Agent
# Includes:
# - Official Firecrawl Local binary built from Rust/crates.io (https://github.com/firecrawl/firecrawl#firecrawl-local)
# - Chromium for Firecrawl Local JavaScript rendering
# - Node.js 22 Gateway + React UI (built dist/)
# - Python 3.11+ Agent-Core in isolated /app/.venv
#
# Tip: If your host Docker bridge has VPN/systemd-resolved DNS issues during standalone builds,
# run: docker build --network=host -t nemofy-agent .
# (`docker compose up --build` already sets build.network: host automatically.)

# Stage 1: Build official `firecrawl-local` binary on Debian Bookworm (matches runtime glibc)
FROM rust:1-slim-bookworm AS firecrawl-builder

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_NET_RETRY=5 \
    CARGO_HTTP_MULTIPLEXING=false \
    CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
    CARGO_PROFILE_RELEASE_LTO=thin \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
    CARGO_BUILD_JOBS=2

RUN printf 'Acquire::Retries "5";\nAcquire::ForceIPv4 "true";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libssl-dev \
        perl \
        pkg-config \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install official firecrawl-local from crates.io (fallback to github.com/firecrawl/firecrawl apps/local)
RUN cargo install firecrawl-local --locked || ( \
        git clone --depth 1 https://github.com/firecrawl/firecrawl.git /tmp/firecrawl \
        && cd /tmp/firecrawl/apps/local \
        && cargo build --release \
        && cp target/release/firecrawl-local /usr/local/cargo/bin/firecrawl-local \
        && rm -rf /tmp/firecrawl \
    ) \
    && /usr/local/cargo/bin/firecrawl-local --version

# Stage 2: Production runtime image (Debian Bookworm matches firecrawl-builder glibc)
FROM node:22-bookworm-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=3000 \
    AGENT_API_PORT=8100 \
    SANDBOX_BACKEND=local_isolated \
    ALLOW_UNSAFE_LOCAL_BASH=false \
    FIRECRAWL_API_URL=http://127.0.0.1:3002 \
    FIRECRAWL_PORT=3002 \
    FIRECRAWL_CLOUD=auto \
    FIRECRAWL_LOCAL_BROWSER=auto \
    FIRECRAWL_LOCAL_AUTOSTART=true \
    FIRECRAWL_CHROME_PATH=/usr/bin/chromium

# Install Python 3 runtime + Chromium + tini (PID 1 reaper) + procps (pkill) + curl (healthcheck)
RUN printf 'Acquire::Retries "5";\nAcquire::ForceIPv4 "true";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        curl \
        fonts-liberation \
        git \
        libssl3 \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        tini \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled official `firecrawl-local` binary and verify both firecrawl-local & Chromium
COPY --from=firecrawl-builder /usr/local/cargo/bin/firecrawl-local /usr/local/bin/firecrawl-local
RUN chmod +x /usr/local/bin/firecrawl-local && firecrawl-local --version && chromium --version

WORKDIR /app

# Install Node dependencies first for optimal Docker layer caching (includes vite & tsx)
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then \
        npm ci --no-audit --no-fund; \
    else \
        npm install --no-audit --no-fund; \
    fi

# Install Python Agent-Core dependencies into an isolated venv
# NOTE: README.md is required by pyproject.toml (`readme = "README.md"`) during setuptools metadata build
COPY pyproject.toml README.md ./
COPY Agent-Core ./Agent-Core
COPY MCPs ./MCPs
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --retries 5 --timeout 60 --upgrade pip setuptools wheel \
    && /app/.venv/bin/pip install --retries 5 --timeout 60 --prefer-binary -e . \
    && /app/.venv/bin/python -c "import uvicorn, fastapi, langgraph, langchain_openai, httpx, dotenv, agent_core"

# Copy remaining source files (host node_modules/.venv/.env/data/workspace excluded via .dockerignore)
# and build the React production bundle without re-running host runtime-dep installers
COPY . .
RUN SKIP_RUNTIME_DEPS_BUILD=1 npm run build \
    && test -f /app/dist/index.html

# Enable production mode after devDependencies + frontend build are complete
ENV NODE_ENV=production

# Ensure runtime directories exist with safe permissions
RUN mkdir -p /app/data /app/data/firecrawl-local /app/workspace /app/Skills /app/.firecrawl/bin

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -fsS http://127.0.0.1:3000/api/health >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./node_modules/.bin/tsx", "server.ts"]
