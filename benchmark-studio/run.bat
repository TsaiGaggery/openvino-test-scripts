@echo off
REM OpenVINO Benchmark Studio — Standalone Launcher (Windows)
cd /d "%~dp0"

echo OpenVINO Benchmark Studio
echo =========================
echo.

python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not installed. Installing...
    pip install flask
)

set PORT=%BENCHMARK_STUDIO_PORT%
if "%PORT%"=="" set PORT=8085
echo Starting server on http://localhost:%PORT%
echo Press Ctrl+C to stop
echo.

python server.py
