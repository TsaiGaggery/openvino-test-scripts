"""
Benchmark Runner for OpenVINO Benchmark Studio

Manages benchmark execution as a subprocess with real-time stdout streaming.
"""

import subprocess
import sys
import json
import shutil
import signal
import tempfile
from pathlib import Path
from datetime import datetime


class BenchmarkRunner:
    """Manages benchmark_devices.py execution"""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.process = None
        self.history_dir = self.project_dir / "benchmark-studio" / "results-history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config: dict):
        """
        Start benchmark_devices.py with the given config.
        Yields stdout lines as they come for SSE streaming.
        """
        if self.is_running():
            yield {"type": "error", "data": "A benchmark is already running"}
            return

        # Write config to a temp file
        fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='benchmark_studio_')
        try:
            with open(fd, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            import os
            os.close(fd)
            raise

        benchmark_script = self.project_dir / "benchmark_devices.py"
        python_cmd = "python" if sys.platform == "win32" else "python3"

        try:
            self.process = subprocess.Popen(
                [python_cmd, str(benchmark_script), '--config', temp_path],
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._get_env(),
            )

            yield {"type": "started", "data": "Benchmark started"}

            for line in iter(self.process.stdout.readline, ''):
                if line:
                    yield {"type": "output", "data": line.rstrip('\n')}

            self.process.wait()
            exit_code = self.process.returncode

            if exit_code == 0:
                self._save_to_history()
                yield {"type": "done", "data": "Benchmark completed successfully", "exit_code": 0}
            else:
                yield {"type": "done", "data": f"Benchmark exited with code {exit_code}", "exit_code": exit_code}

        except Exception as e:
            yield {"type": "error", "data": str(e)}
        finally:
            self.process = None
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def cancel(self) -> dict:
        """Cancel the running benchmark"""
        if not self.is_running():
            return {"status": "error", "message": "No benchmark is running"}

        try:
            if sys.platform == 'win32':
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            return {"status": "success", "message": "Benchmark cancelled"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_env(self):
        """Get environment for the subprocess"""
        import os
        env = os.environ.copy()
        # Ensure UTF-8 output for emoji-heavy benchmark scripts
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        return env

    def _save_to_history(self):
        """Copy benchmark_results.json to history with timestamp"""
        results_path = self.project_dir / "benchmark_results.json"
        if not results_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.history_dir / f"run_{timestamp}.json"
        shutil.copy2(str(results_path), str(dest))
        print(f"Results saved to history: {dest}")

    def get_latest_results(self) -> dict:
        """Load the latest benchmark_results.json"""
        results_path = self.project_dir / "benchmark_results.json"
        if not results_path.exists():
            return None
        with open(results_path, 'r') as f:
            return json.load(f)

    def get_history(self) -> list:
        """List all historical benchmark runs"""
        runs = []
        for f in sorted(self.history_dir.glob("run_*.json"), reverse=True):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                runs.append({
                    "id": f.stem,
                    "filename": f.name,
                    "timestamp": data.get("timestamp", f.stem),
                    "models_tested": data.get("models_tested", []),
                    "devices_tested": data.get("devices_tested", []),
                })
            except Exception:
                pass
        return runs

    def get_history_run(self, run_id: str) -> dict:
        """Load a specific historical run"""
        filepath = self.history_dir / f"{run_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, 'r') as f:
            return json.load(f)
