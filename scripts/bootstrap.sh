#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIRECTORY/runtime-config.sh"

UV_VERSION="0.11.14"
UV_ARCHIVE_SHA256="4333af5c0730d94323a7819bbdf87ce92dd07fc857d67fff0059e0fca31b5c02"

section() {
  printf '\n\033[1m%s\033[0m\n' "$1"
}

fail() {
  printf '\nSetup stopped: %s\n' "$1" >&2
  exit 1
}

BOOTSTRAP_LOCK_DIRECTORY=""
release_bootstrap_lock() {
  if [[ -n "$BOOTSTRAP_LOCK_DIRECTORY" ]]; then
    rm -f "$BOOTSTRAP_LOCK_DIRECTORY/pid"
    rmdir "$BOOTSTRAP_LOCK_DIRECTORY" 2>/dev/null || true
  fi
}

acquire_bootstrap_lock() {
  BOOTSTRAP_LOCK_DIRECTORY="$QWEN_RUNTIME_DIRECTORY/bootstrap.lock"
  if ! mkdir "$BOOTSTRAP_LOCK_DIRECTORY" 2>/dev/null; then
    local owner_pid=""
    if [[ -f "$BOOTSTRAP_LOCK_DIRECTORY/pid" ]]; then
      owner_pid="$(<"$BOOTSTRAP_LOCK_DIRECTORY/pid")"
    fi
    if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
      fail "Another Qwen playground setup is already running (process $owner_pid)."
    fi
    rm -f "$BOOTSTRAP_LOCK_DIRECTORY/pid"
    rmdir "$BOOTSTRAP_LOCK_DIRECTORY" 2>/dev/null || \
      fail "Unable to recover the stale setup lock at $BOOTSTRAP_LOCK_DIRECTORY."
    mkdir "$BOOTSTRAP_LOCK_DIRECTORY"
  fi
  printf '%s\n' "$$" >"$BOOTSTRAP_LOCK_DIRECTORY/pid"
  trap release_bootstrap_lock EXIT HUP INT TERM
}

ensure_apple_silicon() {
  [[ "$(uname -s)" == "Darwin" ]] || fail "The MLX runtime requires macOS."
  [[ "$(uname -m)" == "arm64" ]] || fail "The MLX runtime requires Apple silicon."
  command -v git >/dev/null 2>&1 || fail "Install the Xcode command-line tools before continuing."
  command -v node >/dev/null 2>&1 || fail "Node.js 20.19 or 22.12 and newer is required."
  command -v npm >/dev/null 2>&1 || fail "npm is required."
  command -v curl >/dev/null 2>&1 || fail "curl is required."

  local node_supported
  node_supported="$(node -p 'const [major, minor] = process.versions.node.split(".").map(Number); Number((major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major >= 23)')"
  (( node_supported == 1 )) || fail "Node.js 20.19 or 22.12 and newer is required."
}

resolve_uv() {
  local bundled_uv="$QWEN_RUNTIME_DIRECTORY/tools/uv"
  if [[ -x "$bundled_uv" ]]; then
    if [[ "$("$bundled_uv" --version 2>/dev/null)" == "uv $UV_VERSION"* ]]; then
      printf '%s\n' "$bundled_uv"
      return
    fi
    fail "The project-local uv executable is not the pinned $UV_VERSION release. Remove $bundled_uv and retry."
  fi

  section "Downloading the project-local Python environment manager" >&2
  local archive="$QWEN_RUNTIME_DIRECTORY/downloads/uv-aarch64-apple-darwin.tar.gz"
  local archive_part="$archive.part"
  local extract_directory
  mkdir -p "$(dirname "$archive")" "$(dirname "$bundled_uv")"
  if [[ ! -f "$archive" || \
        "$(shasum -a 256 "$archive" | awk '{print $1}')" != "$UV_ARCHIVE_SHA256" ]]; then
    rm -f "$archive_part"
    curl -fL --retry 5 --retry-all-errors \
      "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" \
      --output "$archive_part"
    [[ "$(shasum -a 256 "$archive_part" | awk '{print $1}')" == "$UV_ARCHIVE_SHA256" ]] || \
      fail "The downloaded uv archive failed checksum verification."
    mv "$archive_part" "$archive"
  fi
  extract_directory="$(mktemp -d "$QWEN_RUNTIME_DIRECTORY/tools/uv-extract.XXXXXX")"
  tar -xzf "$archive" -C "$extract_directory" --strip-components=1
  install -m 755 "$extract_directory/uv" "$bundled_uv.tmp"
  mv "$bundled_uv.tmp" "$bundled_uv"
  rm -rf "$extract_directory"
  printf '%s\n' "$bundled_uv"
}

ensure_python_environment() {
  section "Preparing the project-local MLX environment"
  export UV_CACHE_DIR="$QWEN_RUNTIME_DIRECTORY/cache/uv"
  export UV_PYTHON_INSTALL_DIR="$QWEN_RUNTIME_DIRECTORY/python"
  export UV_PYTHON_INSTALL_BIN="0"
  export UV_TOOL_DIR="$QWEN_RUNTIME_DIRECTORY/tools/uv-tools"
  export PIP_CACHE_DIR="$QWEN_RUNTIME_DIRECTORY/cache/pip"

  UV_BINARY="$(resolve_uv)"
  "$UV_BINARY" python install 3.12
  if [[ ! -x "$QWEN_VENV_DIRECTORY/bin/python" ]]; then
    "$UV_BINARY" venv --python 3.12 "$QWEN_VENV_DIRECTORY"
  fi
  "$UV_BINARY" pip install \
    --python "$QWEN_VENV_DIRECTORY/bin/python" \
    --requirements "$QWEN_PROJECT_ROOT/server/requirements.txt"
}

download_model() {
  local repository="$1"
  local revision="$2"
  local destination="$3"
  local label="$4"

  section "Downloading $label"
  mkdir -p "$destination"
  "$QWEN_VENV_DIRECTORY/bin/hf" download \
    "$repository" \
    --revision "$revision" \
    --local-dir "$destination"
  [[ -s "$destination/config.json" ]] || fail "$label config is missing."
  find "$destination" -maxdepth 1 -name '*.safetensors' -type f -size +1c | grep -q . || \
    fail "$label weights are missing or incomplete."
}

ensure_models() {
  export HF_HOME="$QWEN_RUNTIME_DIRECTORY/cache/huggingface"
  export HF_HUB_CACHE="$QWEN_RUNTIME_DIRECTORY/cache/huggingface/hub"
  export HF_XET_CACHE="$QWEN_RUNTIME_DIRECTORY/cache/huggingface/xet"
  export HF_HUB_ENABLE_HF_TRANSFER="0"

  download_model \
    "$QWEN_TARGET_REPOSITORY" "$QWEN_TARGET_REVISION" \
    "$QWEN_TARGET_DIRECTORY" "Qwen3.8-27B 8-bit target"
  download_model \
    "$QWEN_MTP_REPOSITORY" "$QWEN_MTP_REVISION" \
    "$QWEN_MTP_DIRECTORY" "Qwen3.8 native MTP sidecar"
  download_model \
    "$QWEN_DSPARK_REPOSITORY" "$QWEN_DSPARK_REVISION" \
    "$QWEN_DSPARK_DIRECTORY" "Qwen3.8 DSpark drafter"
}

ensure_frontend() {
  section "Building the browser app"
  export npm_config_cache="$QWEN_RUNTIME_DIRECTORY/cache/npm"

  local dependency_stamp
  dependency_stamp="$QWEN_RUNTIME_DIRECTORY/stamps/npm-$(shasum -a 256 "$QWEN_PROJECT_ROOT/package-lock.json" | awk '{print $1}')"
  mkdir -p "$QWEN_RUNTIME_DIRECTORY/stamps"
  if [[ ! -f "$dependency_stamp" || \
        ! -x "$QWEN_PROJECT_ROOT/node_modules/.bin/react-router" || \
        ! -x "$QWEN_PROJECT_ROOT/node_modules/.bin/eslint" || \
        ! -x "$QWEN_PROJECT_ROOT/node_modules/.bin/vitest" ]]; then
    (
      cd "$QWEN_PROJECT_ROOT"
      npm ci
    )
    rm -f "$QWEN_RUNTIME_DIRECTORY"/stamps/npm-*
    touch "$dependency_stamp"
  fi
  (
    cd "$QWEN_PROJECT_ROOT"
    npm run build
  )
}

main() {
  mkdir -p "$QWEN_RUNTIME_DIRECTORY"
  acquire_bootstrap_lock
  ensure_apple_silicon
  ensure_python_environment
  ensure_models
  ensure_frontend
  section "Setup complete"
}

main "$@"
