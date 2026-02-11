#!/usr/bin/env bash
# Run YOLO-World Plate Counter on macOS or Jetson Nano (Ubuntu)
# Usage: ./run.sh

set -e

echo "[run.sh] Starting ..."
# Get directory where this script lives (works on Mac and Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[run.sh] Working directory: $SCRIPT_DIR"

VENV_DIR="venv"
REQUIREMENTS="requirements.txt"
APP="app.py"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[run.sh] Using existing venv: $VENV_DIR"
fi

# Activate virtual environment (works on both macOS and Linux)
echo "[run.sh] Activating virtual environment ..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Install or upgrade dependencies (no -q so you see progress; can take a while on first run)
echo "[run.sh] Upgrading pip ..."
pip install --upgrade pip
echo "[run.sh] Installing dependencies (this may take several minutes) ..."
pip install -r "$REQUIREMENTS"

# Run the Streamlit app (unbuffered so logs show immediately)
echo "[run.sh] Starting Streamlit app (YOLO-World Plate Counter) ..."
echo "[run.sh] Open the URL below in your browser (e.g. http://localhost:8501)"
echo ""
export PYTHONUNBUFFERED=1
exec streamlit run "$APP" --server.headless true
