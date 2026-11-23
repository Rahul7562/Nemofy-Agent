#!/usr/bin/env bash
set -euo pipefail

# Unified server runtime dependency builder.
# Runs on every `npm run build` so every host server build automatically ensures:
#   1. Official `firecrawl-local` binary (with built-in pdf-inspector + anydoc PDF/Office parsing)
#   2. Chromium browser (for Firecrawl Local JavaScript rendering)
# Fast no-op (<0.2s) when both binaries are already present; set FIRECRAWL_FORCE_REBUILD=1 to force recompile.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${SKIP_RUNTIME_DEPS_BUILD:-0}" = "1" ]; then
  echo "[build-runtime-deps] Skipped via SKIP_RUNTIME_DEPS_BUILD=1"
  exit 0
fi

bash "${ROOT_DIR}/scripts/install-firecrawl-local.sh"
bash "${ROOT_DIR}/scripts/install-chromium.sh"
