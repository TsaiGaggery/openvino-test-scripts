#!/usr/bin/env python3
"""
OpenVINO Benchmark Studio — Flask Backend Server

Serves the web UI and provides REST API + SSE for:
- Device detection
- Model management (search, download, register, delete)
- Benchmark configuration (load/save benchmark.json)
- Benchmark execution (start, cancel, stream output)
- Results viewing (latest, history)
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response

# Project root is passed via environment or defaults to parent directory
PROJECT_DIR = Path(os.environ.get('BENCHMARK_PROJECT_DIR', Path(__file__).parent.parent))
STUDIO_DIR = Path(__file__).parent
PORT = int(os.environ.get('BENCHMARK_STUDIO_PORT', 8085))

# Initialize Flask
app = Flask(__name__, static_folder=str(STUDIO_DIR / 'static'))
app.config['JSON_SORT_KEYS'] = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('benchmark-studio')

# Ensure benchmark-studio modules are importable regardless of cwd
sys.path.insert(0, str(STUDIO_DIR))

# Initialize managers
from device_manager import DeviceManager
from model_manager import ModelManager
from benchmark_runner import BenchmarkRunner

device_manager = DeviceManager()
model_manager = ModelManager(
    models_config_path=str(STUDIO_DIR / 'models_config.json'),
    data_dir=str(PROJECT_DIR),
)
benchmark_runner = BenchmarkRunner(project_dir=str(PROJECT_DIR))


# ─── Static Files ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(str(STUDIO_DIR / 'static'), 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(str(STUDIO_DIR / 'static'), filename)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({
        "status": "ok",
        "project_dir": str(PROJECT_DIR),
        "benchmark_running": benchmark_runner.is_running(),
    })


# ─── Device Detection ───────────────────────────────────────────────────────

@app.route('/api/devices')
def get_devices():
    devices = device_manager.detect_devices()
    return jsonify({"devices": devices})


# ─── Model Management ───────────────────────────────────────────────────────

@app.route('/api/models')
def get_models():
    models = model_manager.get_available_models()
    return jsonify({"models": models})


@app.route('/api/models/search')
def search_models():
    query = request.args.get('q', '').strip()
    limit = min(int(request.args.get('limit', 20)), 50)
    if not query:
        return jsonify({"results": []})
    results = model_manager.search_huggingface(query, limit=limit)
    return jsonify({"results": results})


@app.route('/api/model/download/stream', methods=['POST'])
def download_model_stream():
    data = request.get_json()
    hf_model_id = data.get('model_id', '')
    if not hf_model_id:
        return jsonify({"status": "error", "message": "model_id required"}), 400

    def generate():
        for progress in model_manager.download_model_streaming(hf_model_id):
            yield f"data: {json.dumps(progress)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/model/register-local', methods=['POST'])
def register_local_model():
    data = request.get_json()
    folder_path = data.get('folder_path', '')
    if not folder_path:
        return jsonify({"status": "error", "message": "folder_path required"}), 400
    if not os.path.isabs(folder_path):
        return jsonify({"status": "error", "message": "Path must be absolute"}), 400
    result = model_manager.register_local_model(folder_path)
    return jsonify(result)


@app.route('/api/model/delete', methods=['POST'])
def delete_model():
    data = request.get_json()
    hf_model_id = data.get('model_id', '')
    if not hf_model_id:
        return jsonify({"status": "error", "message": "model_id required"}), 400
    result = model_manager.delete_model(hf_model_id)
    return jsonify(result)


# ─── Benchmark Config ───────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "models": [],
    "benchmark_config": {
        "devices_to_test": ["CPU"],
        "generation_config": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "test_prompts": [
            "What is artificial intelligence and how does it work?"
        ],
        "run_warmup": True,
        "cache_dir": "./ov_cache"
    }
}


def _is_local_path(model_id):
    return (model_id.startswith('/') or model_id.startswith('~')
            or model_id.startswith('./') or model_id.startswith('.\\')
            or (len(model_id) >= 2 and model_id[1] == ':')
            or model_id.startswith('\\\\'))


def _local_model_exists(model_id):
    try:
        p = Path(model_id).expanduser()
        return p.exists() and p.is_dir() and list(p.glob('*.xml')) and list(p.glob('*.bin'))
    except (OSError, PermissionError):
        return False


@app.route('/api/config')
def get_config():
    config_path = PROJECT_DIR / 'benchmark.json'
    if not config_path.exists():
        # Create default config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return jsonify(DEFAULT_CONFIG)
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Filter out local-path models that don't exist on disk
    config['models'] = [
        m for m in config.get('models', [])
        if not _is_local_path(m.get('model_id', '')) or _local_model_exists(m.get('model_id', ''))
    ]
    return jsonify(config)


@app.route('/api/config', methods=['POST'])
def save_config():
    if benchmark_runner.is_running():
        return jsonify({"status": "error", "message": "Cannot save config while benchmark is running"}), 409
    config = request.get_json()
    config_path = PROJECT_DIR / 'benchmark.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "success", "message": "Config saved"})


# ─── Benchmark Execution ────────────────────────────────────────────────────

@app.route('/api/benchmark/start', methods=['POST'])
def start_benchmark():
    if benchmark_runner.is_running():
        return jsonify({"status": "error", "message": "Benchmark already running"}), 409

    data = request.get_json() or {}
    # Use provided config or load from benchmark.json
    config = data.get('config')
    if not config:
        config_path = PROJECT_DIR / 'benchmark.json'
        if not config_path.exists():
            return jsonify({"status": "error", "message": "benchmark.json not found"}), 404
        with open(config_path, 'r') as f:
            config = json.load(f)

    def generate():
        for event in benchmark_runner.start(config):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/benchmark/cancel', methods=['POST'])
def cancel_benchmark():
    result = benchmark_runner.cancel()
    return jsonify(result)


@app.route('/api/benchmark/status')
def benchmark_status():
    return jsonify({"running": benchmark_runner.is_running()})


# ─── Results ─────────────────────────────────────────────────────────────────

@app.route('/api/results/latest')
def get_latest_results():
    results = benchmark_runner.get_latest_results()
    if results is None:
        return jsonify({"error": "No results found"}), 404
    return jsonify(results)


@app.route('/api/results/history')
def get_results_history():
    history = benchmark_runner.get_history()
    return jsonify({"runs": history})


@app.route('/api/results/<run_id>')
def get_history_run(run_id):
    results = benchmark_runner.get_history_run(run_id)
    if results is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(results)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info(f"Project dir: {PROJECT_DIR}")
    logger.info(f"Studio dir: {STUDIO_DIR}")
    logger.info(f"Starting on port {PORT}")

    # Ensure models_config.json exists
    models_config = STUDIO_DIR / 'models_config.json'
    if not models_config.exists():
        with open(models_config, 'w') as f:
            json.dump({"models": []}, f, indent=2)

    app.run(host='127.0.0.1', port=PORT, debug=False, threaded=True)
