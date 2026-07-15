#!/usr/bin/env bash
# Bootstrap Athena ODT on a fresh Apple Silicon Mac: creates the "ODT" conda
# env (see ../environment.yml), installs both apps' dependencies, then starts
# sd_server (native, so it gets real MPS/GPU access — Docker cannot provide
# that on Mac, see sd_server/README.md) and the Django app in the background.
#
# Usage: scripts/setup_mac.sh
# Stop:  scripts/stop.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="ODT"
RUN_DIR="$REPO_ROOT/.run"
mkdir -p "$RUN_DIR"

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

echo "Starting sd_server on :7860..."
nohup python sd_server/app.py --port 7860 > "$RUN_DIR/sd_server.log" 2>&1 &
echo $! > "$RUN_DIR/sd_server.pid"

echo "Starting Django app on :8000..."
# --noreload: runserver's autoreloader forks a child process, which would
# leave the pid file pointing at the wrong (parent) process for stop.sh.
nohup python AthensMT/manage.py runserver --noreload 0.0.0.0:8000 > "$RUN_DIR/app.log" 2>&1 &
echo $! > "$RUN_DIR/app.pid"

sleep 2
echo
echo "Done."
echo "  App:       http://localhost:8000/ODT/   (log: .run/app.log)"
echo "  SD server: http://localhost:7860         (log: .run/sd_server.log)"
echo "  Stop both: scripts/stop.sh"
