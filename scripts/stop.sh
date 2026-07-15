#!/usr/bin/env bash
# Stops the app + sd_server processes started by scripts/setup_mac.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_ROOT/.run"

for name in sd_server app; do
  pid_file="$RUN_DIR/$name.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "Stopped $name (pid $pid)."
    else
      echo "$name (pid $pid) was not running."
    fi
    rm -f "$pid_file"
  else
    echo "No pid file for $name — nothing to stop."
  fi
done
