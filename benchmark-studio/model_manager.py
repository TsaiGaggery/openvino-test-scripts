"""
Model Manager for OpenVINO Benchmark Studio

Handles model searching, downloading, registration, and validation.
Adapted from openvino-web/model_manager.py for benchmark use.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional
from threading import Thread
from queue import Queue, Empty


class ModelManager:
    """Manages LLM models for benchmarking"""

    def __init__(self, models_config_path: str = "models_config.json", data_dir: str = None):
        self.config_path = models_config_path
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.models_config = self._load_config()
        self._cleanup_stale_models()
        self._scan_models()

    def _load_config(self) -> dict:
        config_file = Path(self.config_path)
        if not config_file.exists():
            return {"models": []}
        with open(config_file, 'r') as f:
            return json.load(f)

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.models_config, f, indent=2)

    def _cleanup_stale_models(self):
        """Remove model entries whose paths no longer exist on disk"""
        models = self.models_config.get("models", [])
        original_count = len(models)
        cleaned = []
        for m in models:
            local_path = Path(m["local_path"])
            if not local_path.is_absolute():
                local_path = self.data_dir / local_path
            if not local_path.exists() or not self._is_valid_model(local_path):
                print(f"Removing stale model: {m['name']} ({m['local_path']})")
                continue
            cleaned.append(m)
        if len(cleaned) < original_count:
            self.models_config["models"] = cleaned
            self._save_config()

    def _scan_models(self):
        self.available_models = []
        for model_config in self.models_config.get("models", []):
            local_path = Path(model_config["local_path"])
            if not local_path.is_absolute():
                local_path = self.data_dir / local_path

            if local_path.exists() and self._is_valid_model(local_path):
                status = "ready"
            else:
                status = "not_downloaded"

            self.available_models.append({
                "name": model_config["name"],
                "hf_model_id": model_config["hf_model_id"],
                "local_path": str(local_path),
                "description": model_config.get("description", ""),
                "status": status,
            })

    def _is_valid_model(self, model_path: Path) -> bool:
        xml_files = list(model_path.glob("*.xml"))
        bin_files = list(model_path.glob("*.bin"))
        return len(xml_files) > 0 and len(bin_files) > 0

    def get_available_models(self) -> List[Dict]:
        return self.available_models

    def search_huggingface(self, query: str, limit: int = 20, token: str = None) -> List[Dict]:
        """Search HuggingFace for OpenVINO models"""
        from huggingface_hub import HfApi

        api = HfApi(token=token or None)
        results = []

        try:
            models = api.list_models(
                search=query,
                filter="openvino",
                sort="downloads",
                limit=limit,
            )

            local_ids = {m["hf_model_id"] for m in self.available_models}

            for model in models:
                results.append({
                    "id": model.id,
                    "name": model.id.split("/")[-1] if "/" in model.id else model.id,
                    "author": model.id.split("/")[0] if "/" in model.id else "",
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "tags": model.tags or [],
                    "last_modified": str(model.last_modified) if model.last_modified else "",
                    "is_local": model.id in local_ids,
                })
        except Exception as e:
            print(f"HF search error: {e}")

        return results

    def download_model_streaming(self, hf_model_id: str, token: str = None):
        """Download model with streaming progress updates (yields dicts)"""
        from huggingface_hub import snapshot_download
        import tqdm as tqdm_module

        model_config = None
        for cfg in self.models_config.get("models", []):
            if cfg["hf_model_id"] == hf_model_id:
                model_config = cfg
                break

        if not model_config:
            model_name = hf_model_id.split("/")[-1]
            model_config = {
                "name": model_name,
                "hf_model_id": hf_model_id,
                "local_path": f"models/{model_name}",
                "description": f"Downloaded from HuggingFace: {hf_model_id}",
            }
            self.models_config.setdefault("models", []).append(model_config)
            self._save_config()

        local_path = self.data_dir / model_config["local_path"]
        progress_queue = Queue()

        class ProgressTqdm(tqdm_module.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._last_update = 0

            def display(self, msg=None, pos=None):
                now = time.time()
                if now - self._last_update < 0.5 and self.total and self.n < self.total:
                    return
                self._last_update = now
                if self.total and self.total > 0:
                    progress_queue.put({
                        "status": "downloading",
                        "file": str(self.desc or ""),
                        "progress": self.n / self.total,
                        "downloaded": self.format_sizeof(self.n),
                        "total": self.format_sizeof(self.total),
                    })

            def close(self):
                if self.total and self.total > 0:
                    progress_queue.put({
                        "status": "downloading",
                        "file": str(self.desc or ""),
                        "progress": 1.0,
                        "downloaded": self.format_sizeof(self.total),
                        "total": self.format_sizeof(self.total),
                    })
                super().close()

        def do_download():
            try:
                snapshot_download(
                    repo_id=hf_model_id,
                    local_dir=str(local_path),
                    token=token or None,
                    tqdm_class=ProgressTqdm,
                )
                self._scan_models()
                progress_queue.put({
                    "status": "success",
                    "message": f"Model {model_config['name']} downloaded successfully",
                    "local_path": str(local_path),
                })
            except Exception as e:
                if local_path.exists():
                    shutil.rmtree(local_path, ignore_errors=True)
                progress_queue.put({
                    "status": "error",
                    "message": f"Download failed: {str(e)}",
                })
            finally:
                progress_queue.put(None)

        thread = Thread(target=do_download)
        thread.start()

        while True:
            try:
                item = progress_queue.get(timeout=30)
            except Empty:
                yield {"status": "error", "message": "Download timed out"}
                break
            if item is None:
                break
            yield item

        thread.join(timeout=5)

    def register_local_model(self, folder_path: str) -> Dict:
        """Register a local folder as an OpenVINO model"""
        model_dir = Path(folder_path)

        if not model_dir.exists() or not model_dir.is_dir():
            return {"status": "error", "message": f"Directory not found: {folder_path}"}

        if not self._is_valid_model(model_dir):
            return {"status": "error", "message": "Invalid model: no .xml and .bin files found"}

        abs_path = str(model_dir.resolve())
        for existing in self.models_config.get("models", []):
            existing_path = Path(existing["local_path"])
            if not existing_path.is_absolute():
                existing_path = self.data_dir / existing_path
            if str(existing_path.resolve()) == abs_path:
                return {"status": "error", "message": f"Already registered: {existing['name']}"}

        model_name = model_dir.name
        hf_model_id = f"LOCAL/{model_name}"
        existing_ids = {m["hf_model_id"] for m in self.models_config.get("models", [])}
        if hf_model_id in existing_ids:
            counter = 2
            while f"LOCAL/{model_name}-{counter}" in existing_ids:
                counter += 1
            hf_model_id = f"LOCAL/{model_name}-{counter}"
            model_name = f"{model_name}-{counter}"

        model_config = {
            "name": model_name,
            "hf_model_id": hf_model_id,
            "local_path": abs_path,
            "description": f"Local model: {abs_path}",
        }

        self.models_config.setdefault("models", []).append(model_config)
        self._save_config()
        self._scan_models()

        return {
            "status": "success",
            "message": f"Model '{model_name}' registered",
            "model": {"name": model_name, "hf_model_id": hf_model_id, "local_path": abs_path},
        }

    def delete_model(self, hf_model_id: str) -> Dict:
        """Delete a model and remove from config"""
        # Remove from config
        models = self.models_config.get("models", [])
        found = None
        for m in models:
            if m["hf_model_id"] == hf_model_id:
                found = m
                break

        if not found:
            return {"status": "error", "message": "Model not found"}

        # Only delete files for non-local models
        if not hf_model_id.startswith("LOCAL/"):
            local_path = Path(found["local_path"])
            if not local_path.is_absolute():
                local_path = self.data_dir / local_path
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)

        models.remove(found)
        self._save_config()
        self._scan_models()

        return {"status": "success", "message": "Model removed"}
