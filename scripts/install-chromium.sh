#!/usr/bin/env bash
set -euo pipefail

# Ensures Chrome/Chromium is available alongside Firecrawl Local for JS-rendered pages.
# Per https://github.com/firecrawl/firecrawl#firecrawl-local:
# - Static HTML, PDFs, and Office documents run purely in native Rust (no browser needed).
# - JavaScript-heavy pages, waitFor, and screenshots require Chrome/Chromium.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BROWSERS_DIR="${ROOT_DIR}/.firecrawl/browsers"
mkdir -p "${LOCAL_BROWSERS_DIR}"

find_chrome() {
  if [ -n "${FIRECRAWL_CHROME_PATH:-}" ] && [ -x "${FIRECRAWL_CHROME_PATH}" ]; then
    echo "${FIRECRAWL_CHROME_PATH}"
    return 0
  fi
  if [ -n "${PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH:-}" ] && [ -x "${PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH}" ]; then
    echo "${PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH}"
    return 0
  fi

  for cand in \
    /usr/bin/chromium \
    /usr/bin/chromium-browser \
    /usr/bin/google-chrome \
    /usr/bin/google-chrome-stable \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  do
    if [ -x "${cand}" ]; then
      echo "${cand}"
      return 0
    fi
  done

  # Search local project cache, system Playwright cache, and user Playwright cache
  for root in "${LOCAL_BROWSERS_DIR}" /ms-playwright "${HOME}/.cache/ms-playwright"; do
    if [ -d "${root}" ]; then
      local found=""
      found="$(find "${root}" -maxdepth 4 -type f \( -name "chrome" -o -name "headless_shell" -o -name "chrome-headless-shell" \) -perm -111 2>/dev/null | sort | tail -n 1 || true)"
      if [ -n "${found}" ] && [ -x "${found}" ]; then
        echo "${found}"
        return 0
      fi
    fi
  done

  return 1
}

if EXISTING_CHROME="$(find_chrome)"; then
  echo "[chromium] Ready: ${EXISTING_CHROME}"
  exit 0
fi

echo "[chromium] No Chrome/Chromium binary found — installing Chromium alongside server..."

# 1. On Debian/Ubuntu hosts running as root (Docker/VM builds), install system chromium directly
if [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true update -qq
  DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true install -y --no-install-recommends chromium fonts-liberation ca-certificates
  rm -rf /var/lib/apt/lists/*
  if INSTALLED_CHROME="$(find_chrome)"; then
    echo "[chromium] Installed system Chromium: ${INSTALLED_CHROME}"
    exit 0
  fi
fi

# 2. Portable user-space installation via Playwright Chromium into .firecrawl/browsers (no root required)
if command -v npx >/dev/null 2>&1; then
  echo "[chromium] Downloading Chromium into ${LOCAL_BROWSERS_DIR} via playwright-core..."
  PLAYWRIGHT_BROWSERS_PATH="${LOCAL_BROWSERS_DIR}" npx --yes playwright-core install chromium
  if INSTALLED_CHROME="$(find_chrome)"; then
    echo "[chromium] Installed local Chromium: ${INSTALLED_CHROME}"
    exit 0
  fi
fi

echo "[chromium] WARNING: Could not automatically install Chromium. Install Chromium/Chrome or set FIRECRAWL_CHROME_PATH for JS-rendered pages (static pages, PDFs, and Office docs will still work)." >&2
exit 0
