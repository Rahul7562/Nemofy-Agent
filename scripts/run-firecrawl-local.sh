#!/usr/bin/env bash
set -euo pipefail

# Launch official `firecrawl-local serve` on localhost (default port 3002)
# Reference: https://github.com/firecrawl/firecrawl#firecrawl-local

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${ROOT_DIR}/.firecrawl/bin:${HOME}/.cargo/bin:/usr/local/cargo/bin:${PATH}"

if ! command -v firecrawl-local >/dev/null 2>&1; then
  bash "${ROOT_DIR}/scripts/install-firecrawl-local.sh"
fi

# Auto-discover Chrome/Chromium if FIRECRAWL_CHROME_PATH is not set
if [ -z "${FIRECRAWL_CHROME_PATH:-}" ]; then
  for cand in \
    /usr/bin/chromium \
    /usr/bin/chromium-browser \
    /usr/bin/google-chrome \
    /usr/bin/google-chrome-stable \
    "${ROOT_DIR}"/.firecrawl/browsers/chromium-*/chrome-linux*/chrome \
    /ms-playwright/chromium-*/chrome-linux*/chrome \
    "${HOME}"/.cache/ms-playwright/chromium-*/chrome-linux*/chrome \
    "${ROOT_DIR}"/.firecrawl/browsers/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell \
    /ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell; do
    if [ -x "$cand" ]; then
      export FIRECRAWL_CHROME_PATH="$cand"
      break
    fi
  done
fi

# Inside root Linux containers, wrap Chrome with --no-sandbox --disable-dev-shm-usage
if [ -n "${FIRECRAWL_CHROME_PATH:-}" ] && [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" = "0" ]; then
  WRAPPER="${ROOT_DIR}/.firecrawl/bin/chrome-launcher.sh"
  mkdir -p "${ROOT_DIR}/.firecrawl/bin"
  cat > "${WRAPPER}" <<EOF
#!/usr/bin/env bash
exec "${FIRECRAWL_CHROME_PATH}" --no-sandbox --disable-dev-shm-usage "\$@"
EOF
  chmod +x "${WRAPPER}"
  export FIRECRAWL_CHROME_PATH="${WRAPPER}"
fi

HOST="${FIRECRAWL_LOCAL_HOST:-127.0.0.1}"
PORT="${FIRECRAWL_PORT:-3002}"
export FIRECRAWL_CLOUD="${FIRECRAWL_CLOUD:-auto}"
export FIRECRAWL_LOCAL_BROWSER="${FIRECRAWL_LOCAL_BROWSER:-auto}"
export FIRECRAWL_LOCAL_HOME="${FIRECRAWL_LOCAL_HOME:-${ROOT_DIR}/data/firecrawl-local}"
mkdir -p "${FIRECRAWL_LOCAL_HOME}"

ARGS=("serve" "--host" "${HOST}" "--port" "${PORT}")
if [ -n "${FIRECRAWL_LOCAL_TOKEN:-}" ]; then
  ARGS+=("--token" "${FIRECRAWL_LOCAL_TOKEN}")
fi
if [ "${FIRECRAWL_LOCAL_ALLOW_PRIVATE:-false}" = "true" ] || [ "${FIRECRAWL_LOCAL_ALLOW_PRIVATE:-0}" = "1" ]; then
  ARGS+=("--allow-private")
fi

exec firecrawl-local "${ARGS[@]}"
