#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/runtime-config.sh"
cd "$PROJECT_ROOT"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

printf '\033]0;Qwen3.8 27B\007'
printf '\n\033[1mQwen3.8 27B · Optimized MLX playground\033[0m\n'
printf 'Checking the project-local runtime before launch.\n'

qwen_server_ready() {
  curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
    "http://$QWEN_SERVER_HOST:$QWEN_SERVER_PORT/api/runtime" 2>/dev/null |
    grep -Fq "\"modelId\":\"$QWEN_MODEL_ID\""
}

is_project_backend_pid() {
  local pid="$1"
  local command=""
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == "$QWEN_VENV_DIRECTORY/bin/python -m mlx_dspark "* ||
     "$command" == "$QWEN_VENV_DIRECTORY/bin/python -m mlx_vlm.server "* ]]
}

backend_pid() {
  local pid=""
  if [[ -f "$QWEN_BACKEND_PID_PATH" ]]; then
    pid="$(<"$QWEN_BACKEND_PID_PATH")"
    if is_project_backend_pid "$pid"; then
      printf '%s\n' "$pid"
      return
    fi
    rm -f "$QWEN_BACKEND_PID_PATH"
  fi

  pid="$(lsof -nP -tiTCP:"$QWEN_BACKEND_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if is_project_backend_pid "$pid"; then
    printf '%s\n' "$pid"
  fi
}

stop_project_backend() {
  local pid=""
  pid="$(backend_pid)"
  [[ -n "$pid" ]] || return 0

  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..40}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if is_project_backend_pid "$pid"; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$QWEN_BACKEND_PID_PATH"
}

if qwen_server_ready; then
  open "http://$QWEN_SERVER_HOST:$QWEN_SERVER_PORT"
  printf 'Qwen3.8 is already running. Opened it in your browser.\n'
  exit 0
fi

mkdir -p "$QWEN_RUNTIME_DIRECTORY"
LAUNCH_LOCK_DIRECTORY="$QWEN_RUNTIME_DIRECTORY/launch.lock"
if ! mkdir "$LAUNCH_LOCK_DIRECTORY" 2>/dev/null; then
  LAUNCH_OWNER_PID=""
  if [[ -f "$LAUNCH_LOCK_DIRECTORY/pid" ]]; then
    LAUNCH_OWNER_PID="$(<"$LAUNCH_LOCK_DIRECTORY/pid")"
  fi
  if [[ "$LAUNCH_OWNER_PID" =~ ^[0-9]+$ ]] && kill -0 "$LAUNCH_OWNER_PID" 2>/dev/null; then
    printf 'Qwen3.8 is already starting in another window.\n'
    exit 0
  fi
  rm -f "$LAUNCH_LOCK_DIRECTORY/pid"
  rmdir "$LAUNCH_LOCK_DIRECTORY" 2>/dev/null || {
    printf 'Unable to recover the stale launch lock at %s.\n' "$LAUNCH_LOCK_DIRECTORY" >&2
    exit 1
  }
  mkdir "$LAUNCH_LOCK_DIRECTORY"
fi
printf '%s\n' "$$" >"$LAUNCH_LOCK_DIRECTORY/pid"

CAFFEINATE_PID=""
BROWSER_WAITER_PID=""
SERVER_PID=""
CLEANUP_COMPLETE=0
cleanup() {
  if [[ "$CLEANUP_COMPLETE" -eq 1 ]]; then
    return
  fi
  CLEANUP_COMPLETE=1
  trap - EXIT HUP INT TERM

  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in {1..40}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      SERVER_COMMAND="$(ps -p "$SERVER_PID" -o command= 2>/dev/null || true)"
      if [[ "$SERVER_COMMAND" == "$QWEN_VENV_DIRECTORY/bin/python $PROJECT_ROOT/server/app.py "* ]]; then
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi

  stop_project_backend
  if [[ -n "$BROWSER_WAITER_PID" ]]; then
    kill "$BROWSER_WAITER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$LAUNCH_LOCK_DIRECTORY/pid"
  rmdir "$LAUNCH_LOCK_DIRECTORY" 2>/dev/null || true
}

handle_signal() {
  local exit_status="$1"
  cleanup
  exit "$exit_status"
}

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

ORPHANED_BACKEND_PID="$(backend_pid)"
if [[ -n "$ORPHANED_BACKEND_PID" ]]; then
  printf 'Stopping a leftover Qwen backend from the previous run.\n'
  stop_project_backend
fi

for port in "$QWEN_SERVER_PORT" "$QWEN_BACKEND_PORT"; do
  if nc -z "$QWEN_SERVER_HOST" "$port" >/dev/null 2>&1; then
    printf 'Port %s is already used by a different local app.\n' "$port" >&2
    exit 1
  fi
done

caffeinate -dimsu -w $$ &
CAFFEINATE_PID=$!

"$PROJECT_ROOT/scripts/bootstrap.sh"

open_when_ready() {
  for _ in {1..120}; do
    if qwen_server_ready; then
      open "http://$QWEN_SERVER_HOST:$QWEN_SERVER_PORT"
      return
    fi
    sleep 1
  done
  printf '\nThe playground server did not start. Review the terminal output above.\n' >&2
}

open_when_ready &
BROWSER_WAITER_PID=$!

printf '\nThe playground will open immediately and show model-loading progress.\n'
printf 'Closing this window or pressing Control-C stops Qwen3.8 and frees its memory.\n\n'

"$QWEN_VENV_DIRECTORY/bin/python" "$PROJECT_ROOT/server/app.py" \
  --target-model "$QWEN_TARGET_DIRECTORY" \
  --mtp-model "$QWEN_MTP_DIRECTORY" \
  --dspark-model "$QWEN_DSPARK_DIRECTORY" \
  --python "$QWEN_VENV_DIRECTORY/bin/python" \
  --default-backend "$QWEN_DEFAULT_BACKEND" \
  --max-context "$QWEN_CONTEXT_LENGTH" \
  --max-image-bytes "$QWEN_MAX_IMAGE_BYTES" \
  --backend-port "$QWEN_BACKEND_PORT" \
  --host "$QWEN_SERVER_HOST" \
  --port "$QWEN_SERVER_PORT" &
SERVER_PID=$!

# Keep this small supervisor alive long enough to reap the gateway and remove
# runtime locks after Terminal hangs up. The already-started gateway and model
# backend retain the normal HUP behavior and exit with the Terminal session.
trap '' HUP
wait "$SERVER_PID"
