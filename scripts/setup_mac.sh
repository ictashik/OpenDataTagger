#!/usr/bin/env bash
# Bootstrap Athena ODT on a fresh Apple Silicon Mac: creates the "ODT" conda
# env (see ../environment.yml), installs both apps' dependencies, then starts
# sd_server (native, so it gets real MPS/GPU access — Docker cannot provide
# that on Mac, see sd_server/README.md) and the Django app in the background.
#
# Safe to re-run at any time (e.g. after `git pull`): it stops any instance
# it previously started before starting fresh ones, so it doubles as a
# restart script. It also installs a post-merge git hook so that future
# `git pull`s on this machine auto-restart the app — see scripts/hooks/.
#
# Usage: scripts/setup_mac.sh
# Stop:  scripts/stop.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="ODT"
RUN_DIR="$REPO_ROOT/.run"
mkdir -p "$RUN_DIR"

if [ -d "$REPO_ROOT/.git" ]; then
  for hook in "$REPO_ROOT"/scripts/hooks/*; do
    [ -f "$hook" ] || continue
    install -m 755 "$hook" "$REPO_ROOT/.git/hooks/$(basename "$hook")"
  done
  echo "Installed git hooks (git pull will now auto-restart the app)."
fi

stop_if_running() {
  local name="$1" pid_file="$RUN_DIR/$1.pid" pid
  [ -f "$pid_file" ] || return 0
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping existing $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

# Belt-and-suspenders: kill whatever is actually bound to our ports, not just
# whatever stop_if_running's pid file happens to point at. A pid file goes
# stale the moment a start attempt dies on "Address already in use" (its own
# dead pid overwrites the file) — after that, pid-file-only cleanup can never
# reach the real, orphaned listener again, and every future run silently
# fails to update it while it keeps serving old code.
free_port() {
  local port="$1" pids
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  [ -z "$pids" ] && return 0
  echo "Port $port is still in use by pid(s) $pids — stopping..."
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
}

stop_if_running sd_server
stop_if_running app
free_port 7860
free_port 8000

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Conda env '$ENV_NAME' already exists — skipping creation."
else
  echo "Creating conda env '$ENV_NAME' from environment.yml..."
  conda env create -f environment.yml
fi

conda activate "$ENV_NAME"

echo "Installing Python dependencies..."
pip install --no-cache-dir -r AthensMT/requirements.txt -r sd_server/requirements.txt

if [ ! -f sd_server/.env ] && [ -f sd_server/.env.example ]; then
  cp sd_server/.env.example sd_server/.env
  echo "Created sd_server/.env — add your Hugging Face token there for gated models (SD 3.5, FLUX.1-dev)."
fi

echo "Applying Django migrations..."
python AthensMT/manage.py migrate --noinput

check_started() {
  local name="$1" port="$2" pid_file="$RUN_DIR/$1.pid" pid
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name (pid $pid) died immediately — check $RUN_DIR/$name.log" >&2
    tail -n 20 "$RUN_DIR/$name.log" >&2 || true
    exit 1
  fi
  if [ -z "$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)" ]; then
    echo "ERROR: $name (pid $pid) is running but nothing is listening on :$port — check $RUN_DIR/$name.log" >&2
    exit 1
  fi
}

echo "Starting sd_server on :7860..."
nohup python sd_server/app.py --port 7860 > "$RUN_DIR/sd_server.log" 2>&1 &
echo $! > "$RUN_DIR/sd_server.pid"

echo "Starting Django app on :8000..."
# --noreload: runserver's autoreloader forks a child process, which would
# leave the pid file pointing at the wrong (parent) process for stop.sh.
nohup python AthensMT/manage.py runserver --noreload 0.0.0.0:8000 > "$RUN_DIR/app.log" 2>&1 &
echo $! > "$RUN_DIR/app.pid"

sleep 2
check_started sd_server 7860
check_started app 8000
echo
echo "Done."
echo "  App:       http://localhost:8000/ODT/   (log: .run/app.log)"
echo "  SD server: http://localhost:7860         (log: .run/sd_server.log)"
echo "  Stop both: scripts/stop.sh"
