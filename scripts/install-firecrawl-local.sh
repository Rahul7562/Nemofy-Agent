#!/usr/bin/env bash
set -euo pipefail

# Official Firecrawl Local installer
# Strictly follows https://github.com/firecrawl/firecrawl#firecrawl-local
# Installs Rust toolchain (if missing) + builds/installs `firecrawl-local` from crates.io
# (with fallback to building `apps/local` directly from github.com/firecrawl/firecrawl).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN_DIR="${ROOT_DIR}/.firecrawl/bin"
LOCAL_BIN="${LOCAL_BIN_DIR}/firecrawl-local"
mkdir -p "${LOCAL_BIN_DIR}"

export PATH="${LOCAL_BIN_DIR}:${HOME}/.cargo/bin:/usr/local/cargo/bin:${PATH}"

if [ "${FIRECRAWL_FORCE_REBUILD:-0}" != "1" ] && command -v firecrawl-local >/dev/null 2>&1; then
  EXISTING_BIN="$(command -v firecrawl-local)"
  if [ "${EXISTING_BIN}" != "${LOCAL_BIN}" ]; then
    ln -sf "${EXISTING_BIN}" "${LOCAL_BIN}"
  fi
  echo "[firecrawl-local] Already installed: $("${LOCAL_BIN}" --version)"
  exit 0
fi

# 1. Verify C compiler is present (required by firecrawl-local / mimalloc)
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  echo "[firecrawl-local] ERROR: C compiler (cc/gcc) not found. Install build-essential (Debian/Ubuntu) or Xcode CLI tools (macOS)." >&2
  exit 1
fi

# 2. Ensure Rust / Cargo toolchain is available
if ! command -v cargo >/dev/null 2>&1; then
  echo "[firecrawl-local] Rust/Cargo not found — installing minimal stable Rust toolchain via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

echo "[firecrawl-local] Using $(rustc --version) / $(cargo --version)"

# 3. Install official `firecrawl-local` binary per https://github.com/firecrawl/firecrawl#firecrawl-local
TARGET_DIR="${ROOT_DIR}/.firecrawl/target"
FORCE_FLAG=()
if [ "${FIRECRAWL_FORCE_REBUILD:-0}" = "1" ]; then
  FORCE_FLAG=(--force)
fi
if cargo install firecrawl-local --locked "${FORCE_FLAG[@]}" --target-dir "${TARGET_DIR}"; then
  INSTALLED_BIN="$(command -v firecrawl-local || echo "${HOME}/.cargo/bin/firecrawl-local")"
  ln -sf "${INSTALLED_BIN}" "${LOCAL_BIN}"
  rm -rf "${TARGET_DIR}"
  echo "[firecrawl-local] Installed from crates.io: $("${LOCAL_BIN}" --version)"
  exit 0
fi

# 4. Fallback: clone https://github.com/firecrawl/firecrawl and build apps/local directly
echo "[firecrawl-local] crates.io install failed — building apps/local from https://github.com/firecrawl/firecrawl..."
TMP_BUILD_DIR="$(mktemp -d /tmp/firecrawl-local-build-XXXXXX)"
trap 'rm -rf "${TMP_BUILD_DIR}"' EXIT

git clone --depth 1 https://github.com/firecrawl/firecrawl.git "${TMP_BUILD_DIR}/firecrawl"
(
  cd "${TMP_BUILD_DIR}/firecrawl/apps/local"
  cargo build --release
)
cp "${TMP_BUILD_DIR}/firecrawl/apps/local/target/release/firecrawl-local" "${LOCAL_BIN}"
chmod +x "${LOCAL_BIN}"
echo "[firecrawl-local] Built from GitHub source: $("${LOCAL_BIN}" --version)"
