"""Local workbench state; independent of the browser or a future desktop shell."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_bench.api.benchmark import run_benchmark
from llm_bench.api.config import BenchmarkConfig
from llm_bench.api.credentials import KEY_VARIABLES, read_env_file, resolve_api_key
from llm_bench.api.protocol import DEFAULT_URLS
from llm_bench.api.suite import description


class Workbench:
    def __init__(self, output_dir: Path, state_dir: Path, env_file: Path):
        self.output_dir = output_dir.resolve()
        self.state_dir = state_dir.resolve()
        self.env_file = env_file
        self.lock = threading.Lock()
        self.job: dict[str, Any] | None = None
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None

    def bootstrap(self) -> dict[str, Any]:
        env_values = read_env_file(self.env_file)
        return {
            "defaults": asdict(BenchmarkConfig()),
            "suite": description(),
            "providers": [
                {"id": provider, "baseUrl": DEFAULT_URLS.get(provider, ""),
                 "keyConfigured": bool(os.environ.get(variable) or env_values.get(variable))}
                for provider, variable in KEY_VARIABLES.items()
            ],
            "profiles": self.profiles(),
            "outputDir": str(self.output_dir),
        }

    def profiles(self) -> dict[str, Any]:
        path = self.state_dir / "profiles.json"
        if not path.exists():
            return {}
        profiles = json.loads(path.read_text(encoding="utf-8"))
        # Explicit migration of v1 profiles: keep connections, retire workload knobs.
        retired = {
            "prompt_chars", "repetitions", "concurrency", "probe_output_tokens",
            "decode_output_tokens", "decode_prompt_chars", "timeout", "max_requests",
            "max_prompt_chars", "max_output_tokens",
        }
        return {
            name: asdict(BenchmarkConfig.from_dict({key: value for key, value in values.items() if key not in retired}))
            for name, values in profiles.items()
        }

    def save_profile(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Profile name must contain 1–100 characters")
        values = asdict(BenchmarkConfig.from_dict(config))
        # Configuration accepts no credential fields, including nested ones.
        with self.lock:
            profiles = self.profiles()
            profiles[name.strip()] = values
            self.state_dir.mkdir(parents=True, exist_ok=True)
            target = self.state_dir / "profiles.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
        return profiles

    def start(self, values: dict[str, Any], api_key: str | None, use_saved_key: bool) -> dict[str, Any]:
        config = BenchmarkConfig.from_dict(values)
        if api_key is not None and (not isinstance(api_key, str) or len(api_key) > 16384):
            raise ValueError("Invalid session key")
        if type(use_saved_key) is not bool:
            raise ValueError("Invalid credential selection")
        key = resolve_api_key(config.provider, None, self.env_file) if use_saved_key else api_key
        config.validate_key(key)
        with self.lock:
            if self.job and self.job["status"] in {"running", "stopping"}:
                raise ValueError("A benchmark is already running")
            self.stop = threading.Event()
            self.job = {
                "id": uuid.uuid4().hex, "status": "running", "error": "",
                "completed": 0, "total": config.plan()["requests"],
                "events": [], "resultFile": None,
            }
            self.worker = threading.Thread(target=self._execute, args=(config, key), daemon=True)
            self.worker.start()
            return self._snapshot()

    def _execute(self, config: BenchmarkConfig, key: str | None) -> None:
        try:
            payload = run_benchmark(
                **config.run_options(), api_key=key,
                cancel=self.stop, on_progress=self._progress,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.lock:
                job_id = self.job["id"]
            name = f"llm-api-{payload['createdAt'][:19].replace(':', '-')}-{job_id[:8]}.json"
            target = self.output_dir / name
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
            with self.lock:
                self.job.update(status=payload["runStatus"], resultFile=name)
        except Exception as exc:
            # Task failures cross the worker boundary as safe status, never provider bodies or keys.
            message = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            with self.lock:
                self.job.update(status="failed", error=message)

    def _progress(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.job["completed"] = event["completed"]
            self.job["events"].append(event)

    def _snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.job))

    def status(self) -> dict[str, Any] | None:
        with self.lock:
            return self._snapshot()

    def cancel(self) -> dict[str, Any] | None:
        with self.lock:
            if self.job and self.job["status"] == "running":
                self.job["status"] = "stopping"
                self.stop.set()
            return self._snapshot()

    def history(self) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.output_dir.glob("llm-api-*.json"), reverse=True):
            payload = self.result(path.name)["payload"]
            if payload.get("kind") != "llm-api":
                continue
            entries.append({
                "file": path.name, "createdAt": payload["createdAt"],
                "config": payload["config"], "summary": payload["summary"],
                "status": payload.get("runStatus", "completed"),
            })
        return entries

    def result(self, filename: str) -> dict[str, Any]:
        if not re.fullmatch(r"llm-api-[A-Za-z0-9_.-]+\.json", filename):
            raise ValueError("Invalid result filename")
        target = (self.output_dir / filename).resolve()
        if target.parent != self.output_dir:
            raise ValueError("Result must be inside the output directory")
        return {"payload": json.loads(target.read_text(encoding="utf-8")), "path": str(target)}
