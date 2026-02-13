#!/bin/bash
# OpenVINO Benchmark Studio — Standalone Launcher
cd "$(dirname "$0")"

echo "OpenVINO Benchmark Studio"
echo "========================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Check Flask
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not installed. Installing..."
    pip install flask
fi

PORT=${BENCHMARK_STUDIO_PORT:-8085}
echo "Starting server on http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

python3 server.py
