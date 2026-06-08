from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from llm_bench.runtimes.base import RuntimeConfig, RuntimeResult


class LlamaCppRuntime:
    name = "llama.cpp"

    def __init__(self, executable: str | None = None, mock_result_file: str | None = None):
        self.executable = executable
        self.mock_result_file = mock_result_file

    def run(self, config: RuntimeConfig) -> RuntimeResult:
        rows = self._load_rows(config)
        return self._build_result(config, rows)

    def _load_rows(self, config: RuntimeConfig) -> List[Dict[str, Any]]:
        if self.mock_result_file:
            with Path(self.mock_result_file).expanduser().open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Mock llama-bench result must be a JSON array.")
            return payload

        executable = self.executable or shutil.which("llama-bench")
        if not executable:
            raise FileNotFoundError("llama-bench executable was not found. Pass --llama-bench.")

        command = [
            executable,
            "-m",
            config.model_path,
            "-p",
            str(config.prompt_tokens),
            "-n",
            str(config.generation_tokens),
            "-b",
            str(config.batch_size),
            "-r",
            str(config.repetitions),
            "-ngl",
            str(config.n_gpu_layers),
            "-dev",
            config.device,
            "-o",
            "json",
        ]
        if config.threads is not None:
            command.extend(["-t", str(config.threads)])

        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list):
            raise ValueError("llama-bench JSON output must be an array.")
        return payload

    def _build_result(self, config: RuntimeConfig, rows: List[Dict[str, Any]]) -> RuntimeResult:
        pp_row = _select_pp_row(rows, config.prompt_tokens)
        tg_row = _select_tg_row(rows, config.generation_tokens)
        anchor = pp_row or tg_row or {}

        return RuntimeResult(
            model=config.model,
            baseModel=config.base_model,
            artifact=config.artifact,
            runtime=self.name,
            runtimeVersion=_format_build(anchor),
            accelerationBackend=str(anchor.get("backends", "")),
            promptTokens=config.prompt_tokens,
            generationTokens=config.generation_tokens,
            batchSize=int(anchor.get("n_batch") or config.batch_size),
            repetitions=config.repetitions,
            ppTps=_metric(pp_row, "avg_ts"),
            ppStddev=_metric(pp_row, "stddev_ts"),
            tgTps=_metric(tg_row, "avg_ts"),
            tgStddev=_metric(tg_row, "stddev_ts"),
            nGpuLayers=int(anchor.get("n_gpu_layers") or config.n_gpu_layers),
            threads=_int_or_none(anchor.get("n_threads")) or config.threads,
            backendRaw=str(anchor.get("backends", "")),
            modelSizeBytes=_int_or_none(anchor.get("model_size")),
            modelParams=_int_or_none(anchor.get("model_n_params")),
            rawResult={"llamaBench": rows},
            note="",
            date="",
        )


def _select_pp_row(rows: List[Dict[str, Any]], prompt_tokens: int) -> Dict[str, Any] | None:
    for row in rows:
        if int(row.get("n_prompt") or 0) == prompt_tokens and int(row.get("n_gen") or 0) == 0:
            return row
    return None


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


def _format_build(row: Dict[str, Any]) -> str:
    commit = row.get("build_commit")
    number = row.get("build_number")
    if commit and number:
        return f"{commit} ({number})"
    if commit:
        return str(commit)
    return ""
