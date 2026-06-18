from __future__ import annotations

import argparse
import json

from llm_bench.config import load_config, resolve_model_path, selected_cases
from llm_bench.payload import build_payload, encode_payload
from llm_bench.runtimes.base import RuntimeConfig, RuntimeResult
from llm_bench.runtimes.llama_cpp import LlamaCppRuntime
from scripts.probe_benchmark_env import probe_benchmark_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone LLM inference benchmark launcher.")
    parser.add_argument("--config", help="Path to LLM benchmark config JSON. Defaults to llm_bench/configs/default.json.")
    parser.add_argument("--runtime", choices=["llama.cpp"], default="llama.cpp")
    parser.add_argument("--case", action="append", help="Case name to run. Repeat to run multiple cases. Defaults to all configured cases.")
    parser.add_argument("--list-cases", action="store_true", help="List configured cases and exit.")
    parser.add_argument("--model-path", help="Path to the GGUF model file. Defaults to the configured localPath.")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable.")
    parser.add_argument("--mock-result-file", help="Read llama-bench JSON output from this file.")
    parser.add_argument("-b", "--batch-size", type=int, help="llama-bench batch size. Defaults to configured value.")
    parser.add_argument("-r", "--repetitions", type=int, help="llama-bench repetitions. Defaults to configured value.")
    parser.add_argument("-dev", "--device", help="llama-bench device selector. Defaults to configured value.")
    parser.add_argument("-t", "--threads", type=int)
    parser.add_argument("--backend", default="auto", help="Host metadata backend probe.")
    parser.add_argument("--gpu-id", default="0", help="Comma-separated device ids for metadata probing.")
    parser.add_argument("--note", default="")
    parser.add_argument("--pretty", action="store_true", help="Print decoded JSON before the payload line.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bench_config = load_config(args.config)
    cases = selected_cases(bench_config, args.case)

    if args.list_cases:
        for case in cases:
            print(
                f"{case['name']}: p={case['promptTokens']} g={case['generationTokens']} "
                f"tier={case.get('tier', 'default')}"
            )
        return 0

    runtime = LlamaCppRuntime(
        executable=args.llama_bench,
        mock_result_file=args.mock_result_file,
    )
    model = bench_config["model"]
    runtime_config = bench_config["runtime"]
    defaults = bench_config["defaults"]
    model_path = args.model_path or str(resolve_model_path(bench_config))
    results = []

    for case in cases:
        config = RuntimeConfig(
            case_name=case["name"],
            case_description=case.get("description", ""),
            model=model["key"],
            model_path=model_path,
            base_model=model["baseModel"],
            artifact=model["artifact"],
            prompt_tokens=int(case["promptTokens"]),
            generation_tokens=int(case["generationTokens"]),
            batch_size=args.batch_size or int(defaults["batchSize"]),
            repetitions=args.repetitions or int(defaults["repetitions"]),
            n_gpu_layers=int(runtime_config["nGpuLayers"]),
            device=args.device or runtime_config["device"],
            threads=args.threads if args.threads is not None else defaults.get("threads"),
        )
        try:
            results.append(runtime.run(config))
        except Exception as exc:
            results.append(_failed_result(config, str(exc)))

    host = probe_benchmark_env(requested_backend=args.backend, device_ids=_parse_device_ids(args.gpu_id))
    payload = build_payload(host=host, results=results, note=args.note)

    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(encode_payload(payload))
    return 0


def _parse_device_ids(value: str) -> list[int]:
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [int(chunk) for chunk in chunks] or [0]


def _failed_result(config: RuntimeConfig, error: str) -> RuntimeResult:
    return RuntimeResult(
        caseName=config.case_name,
        caseDescription=config.case_description,
        status="failed",
        error=error.strip(),
        model=config.model,
        baseModel=config.base_model,
        artifact=config.artifact,
        runtime="llama.cpp",
        runtimeVersion="",
        accelerationBackend="",
        promptTokens=config.prompt_tokens,
        generationTokens=config.generation_tokens,
        batchSize=config.batch_size,
        repetitions=config.repetitions,
        ppTps=None,
        ppStddev=None,
        tgTps=None,
        tgStddev=None,
        nGpuLayers=config.n_gpu_layers,
        threads=config.threads,
        backendRaw="",
        modelSizeBytes=None,
        modelParams=None,
        rawResult={},
        note="",
        date="",
    )


if __name__ == "__main__":
    raise SystemExit(main())
