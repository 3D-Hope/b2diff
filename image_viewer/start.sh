#!/bin/bash
# Starts Flask API (port 5005) and Vite dev server (port 5173)
# Port-forward from  local machine: ssh -L 5173:localhost:5173 user@server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python deps..."
pip install flask flask-cors -q

echo "Installing npm deps..."
cd "$SCRIPT_DIR"
npm install

echo ""
echo "Starting Flask API on :5005 ..."
python "$SCRIPT_DIR/server.py" &
FLASK_PID=$!

echo "Starting Vite on :5173 ..."
npm run dev &
VITE_PID=$!

echo ""
echo "============================================"
echo "  PORT FORWARD:  ssh -L 5173:localhost:5173 user@REMOTE"
echo "  Then open:     http://localhost:5173"
echo "============================================"

trap "kill $FLASK_PID $VITE_PID 2>/dev/null" EXIT INT TERM
wait
