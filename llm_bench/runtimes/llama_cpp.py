from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

from llm_bench.runtimes.base import RuntimeConfig, RuntimeResult

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_BENCH_CANDIDATES = (
    ROOT_DIR / "third_party/llama-bench/current/bin/llama-bench",
    ROOT_DIR / "third_party/llama.cpp/build/bin/llama-bench",
    ROOT_DIR / "third_party/llama.cpp/build/bin/Release/llama-bench",
    ROOT_DIR / "third_party/llama.cpp/build/bin/llama-bench.exe",
    ROOT_DIR / "third_party/llama.cpp/build/bin/Release/llama-bench.exe",
)


class LlamaCppRuntime:
    name = "llama.cpp"

    def __init__(self, executable: str | None = None, mock_result_file: str | None = None):
        self.executable = executable
        self.mock_result_file = mock_result_file
        self._last_command: List[str] | None = None

    def run(self, config: RuntimeConfig) -> RuntimeResult:
        rows = self._load_rows(config)
        return self._build_result(config, rows)

    def resolve_executable(self) -> str | None:
        if self.executable:
            return str(Path(self.executable).expanduser())

        env_executable = os.environ.get("GPU_INSIGHTS_LLAMA_BENCH")
        if env_executable:
            return str(Path(env_executable).expanduser())

        for candidate in DEFAULT_LLAMA_BENCH_CANDIDATES:
            if candidate.is_file():
                return str(candidate)

        return shutil.which("llama-bench")

    def _load_rows(self, config: RuntimeConfig) -> List[Dict[str, Any]]:
        self._last_command = None
        if self.mock_result_file:
            with Path(self.mock_result_file).expanduser().open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Mock llama-bench result must be a JSON array.")
            return payload

        executable = self.resolve_executable()
        if not executable:
            default_path = DEFAULT_LLAMA_BENCH_CANDIDATES[0]
            raise FileNotFoundError(
                "llama-bench executable was not found. "
                f"Run scripts/bootstrap-llama-cpp.sh, place it at {default_path}, "
                "set GPU_INSIGHTS_LLAMA_BENCH, make llama-bench available on PATH, "
                "or pass --llama-bench."
            )

        command = [
            executable,
            "-m",
            config.model_path,
            "-p",
            str(config.prompt_tokens),
            "-n",
            str(config.generation_tokens),
            "-d",
            str(_context_depth(config)),
            "-b",
            str(config.batch_size),
            "-r",
            str(config.repetitions),
            "-ngl",
            str(config.n_gpu_layers),
            "-sm",
            config.split_mode,
            "-dev",
            config.device,
            "-o",
            "json",
        ]
        if config.ubatch_size is not None:
            command.extend(["-ub", str(config.ubatch_size)])
        if config.threads is not None:
            command.extend(["-t", str(config.threads)])
        if config.cache_type_k:
            command.extend(["-ctk", config.cache_type_k])
        if config.cache_type_v:
            command.extend(["-ctv", config.cache_type_v])
        if config.flash_attention:
            command.extend(["-fa", "1"])

        self._last_command = command
        stdout = _run_llama_bench(command)
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            preview = stdout.strip()
            if len(preview) > 1000:
                preview = preview[:1000] + "..."
            raise ValueError(f"Failed to parse llama-bench JSON output: {preview}") from exc
        if not isinstance(payload, list):
            raise ValueError("llama-bench JSON output must be an array.")
        return payload

    def _build_result(self, config: RuntimeConfig, rows: List[Dict[str, Any]]) -> RuntimeResult:
        pp_row = _select_pp_row(rows, config.prompt_tokens)
        tg_row = _select_tg_row(rows, config.generation_tokens)
        anchor = pp_row or tg_row or {}

        return RuntimeResult(
            caseName=config.case_name,
            caseDescription=config.case_description,
            status="ok",
            error="",
            model=config.model,
            baseModel=config.base_model,
            artifact=config.artifact,
            runtime=self.name,
            runtimeVersion=_format_build(anchor),
            accelerationBackend=str(anchor.get("backends", "")),
            promptTokens=config.prompt_tokens,
            generationTokens=config.generation_tokens,
            contextSize=config.context_size,
            batchSize=int(anchor.get("n_batch") or config.batch_size),
            ubatchSize=_int_or_none(anchor.get("n_ubatch")) or config.ubatch_size,
            repetitions=config.repetitions,
            ppTps=_metric(pp_row, "avg_ts"),
            ppStddev=_metric(pp_row, "stddev_ts"),
            tgTps=_metric(tg_row, "avg_ts"),
            tgStddev=_metric(tg_row, "stddev_ts"),
            nGpuLayers=int(anchor.get("n_gpu_layers") or config.n_gpu_layers),
            deviceIds=config.device_ids,
            llamaDevices=_format_devices(anchor, config.device),
            splitMode=str(anchor.get("split_mode") or config.split_mode),
            heterogeneousDevices=config.heterogeneous_devices,
            threads=_int_or_none(anchor.get("n_threads")) or config.threads,
            backendRaw=str(anchor.get("backends", "")),
            cacheTypeK=str(anchor.get("type_k") or config.cache_type_k or ""),
            cacheTypeV=str(anchor.get("type_v") or config.cache_type_v or ""),
            flashAttention=bool(anchor.get("flash_attn", config.flash_attention)),
            modelSizeBytes=_int_or_none(anchor.get("model_size")),
            modelParams=_int_or_none(anchor.get("model_n_params")),
            rawResult={
                "llamaBench": rows,
                "llamaBenchCommand": self._last_command,
                "profile": {
                    "contextSize": config.context_size,
                    "contextDepth": _context_depth(config),
                    "batchSize": config.batch_size,
                    "ubatchSize": config.ubatch_size,
                    "deviceIds": config.device_ids,
                    "llamaDevices": config.device,
                    "splitMode": config.split_mode,
                    "heterogeneousDevices": config.heterogeneous_devices,
                    "cacheTypeK": config.cache_type_k,
                    "cacheTypeV": config.cache_type_v,
                    "flashAttention": config.flash_attention,
                },
            },
            note="",
            date="",
        )


def _select_pp_row(rows: List[Dict[str, Any]], prompt_tokens: int) -> Dict[str, Any] | None:
    for row in rows:
        if int(row.get("n_prompt") or 0) == prompt_tokens and int(row.get("n_gen") or 0) == 0:
            return row
    return None


def _run_llama_bench(command: List[str]) -> str:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: List[str] = []

    def stream_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)
            sys.stderr.write(f"  llama-bench | {line}")
            sys.stderr.flush()

    stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
    stderr_thread.start()

    assert process.stdout is not None
    stdout = process.stdout.read()
    returncode = process.wait()
    stderr_thread.join()

    if returncode != 0:
        stderr_tail = "".join(stderr_lines[-20:]).strip()
        message = stderr_tail or stdout.strip() or f"llama-bench exited with code {returncode}"
        raise RuntimeError(message)

    return stdout


def _select_tg_row(rows: List[Dict[str, Any]], generation_tokens: int) -> Dict[str, Any] | None:
    for row in rows:
        if int(row.get("n_prompt") or 0) == 0 and int(row.get("n_gen") or 0) == generation_tokens:
            return row
    return None


def _metric(row: Dict[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _context_depth(config: RuntimeConfig) -> int:
    return max(config.context_size - config.prompt_tokens, 0)


def _format_devices(row: Dict[str, Any], fallback: str) -> str:
    value = row.get("devices")
    if isinstance(value, list):
        return "/".join(str(item) for item in value)
    if value:
        return str(value)
    return fallback


def _format_build(row: Dict[str, Any]) -> str:
    commit = row.get("build_commit")
    number = row.get("build_number")
    if commit and number:
        return f"{commit} ({number})"
    if commit:
        return str(commit)
    return ""
