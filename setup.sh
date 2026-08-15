#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_PATH="$PROJECT_ROOT/Qwen3.8 27B.command"
LAUNCHER_TEMP=""

cleanup() {
  if [[ -n "$LAUNCHER_TEMP" ]]; then
    rm -f "$LAUNCHER_TEMP"
  fi
}
trap cleanup EXIT HUP INT TERM

write_launcher() {
  LAUNCHER_TEMP="$(mktemp "$PROJECT_ROOT/.qwen-launcher.XXXXXX")"
  # shellcheck disable=SC2016
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    '' \
    'PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"' \
    'exec "$PROJECT_ROOT/scripts/launch.sh" "$@"' \
    >"$LAUNCHER_TEMP"
  chmod 755 "$LAUNCHER_TEMP"
  mv -f "$LAUNCHER_TEMP" "$LAUNCHER_PATH"
  LAUNCHER_TEMP=""
}

cd "$PROJECT_ROOT"
printf '\n\033[1mQwen3.8 27B playground setup\033[0m\n'
printf 'This prepares a private MLX runtime and downloads about 34 GB.\n'

"$PROJECT_ROOT/scripts/bootstrap.sh"
write_launcher

printf '\n\033[1mLauncher ready\033[0m\n'
printf 'Double-click %s to open the playground.\n' "$(basename "$LAUNCHER_PATH")"
printf 'Location: %s\n' "$LAUNCHER_PATH"
