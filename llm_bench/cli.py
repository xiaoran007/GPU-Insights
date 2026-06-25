from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from llm_bench.config import load_config, resolve_config_path, resolve_model_path, selected_cases
from llm_bench.payload import build_payload, encode_payload
from llm_bench.runtimes.base import RuntimeConfig, RuntimeResult
from llm_bench.runtimes.llama_cpp import LlamaCppRuntime
from scripts.probe_benchmark_env import probe_benchmark_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone LLM inference benchmark launcher.")
    parser.add_argument("--config", help="Path to LLM benchmark config JSON. Defaults to llm_bench/configs/default.json.")
    parser.add_argument("--gemma", action="store_true", help="Use a non-dashboard Gemma small-GPU preset. Requires --12b or --e2b.")
    gemma_size = parser.add_mutually_exclusive_group()
    gemma_size.add_argument("--12b", dest="gemma_variant", action="store_const", const="12b", help="Use the Gemma 4 12B QAT UD-Q4_K_XL preset.")
    gemma_size.add_argument("--e2b", dest="gemma_variant", action="store_const", const="e2b", help="Use the Gemma 4 E2B QAT UD-Q4_K_XL preset.")
    parser.add_argument("--runtime", choices=["llama.cpp"], default="llama.cpp")
    parser.add_argument("--case", action="append", help="Case name to run. Repeat to run multiple cases. Defaults to all configured cases.")
    parser.add_argument("--list-cases", action="store_true", help="List configured cases and exit.")
    parser.add_argument("--model-path", help="Path to the GGUF model file. Defaults to the configured localPath.")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable.")
    parser.add_argument("--docker", action="store_true", help="Run prebuilt llama-bench inside a CUDA Docker runtime image.")
    parser.add_argument("--mock-result-file", help="Read llama-bench JSON output from this file.")
    parser.add_argument("-b", "--batch-size", type=int, help="llama-bench batch size. Defaults to configured value.")
    parser.add_argument("-ub", "--ubatch-size", type=int, help="llama-bench physical batch size. Defaults to configured value.")
    parser.add_argument("-r", "--repetitions", type=int, help="llama-bench repetitions. Defaults to configured value.")
    parser.add_argument("-dev", "--device", help="llama-bench device selector. Defaults to configured value or CUDA ids.")
    parser.add_argument("-sm", "--split-mode", choices=["none", "layer", "row", "tensor"], help="llama-bench split mode. Defaults to configured value.")
    parser.add_argument("-t", "--threads", type=int)
    parser.add_argument("--backend", default="auto", help="Host metadata backend probe.")
    parser.add_argument("--gpu-id", help="Comma-separated CUDA device ids for both llama-bench execution and metadata probing.")
    parser.add_argument("--note", default="")
    parser.add_argument("--output-dir", default="outputs/llm-bench", help="Directory for payload JSON files.")
    parser.add_argument("--output-file", help="Write the dashboard import payload to this JSON file.")
    parser.add_argument("--debug-output-file", help="Write the full debug payload with raw llama-bench rows to this JSON file.")
    parser.add_argument("--no-debug-output", action="store_true", help="Do not write the full debug payload sidecar.")
    parser.add_argument("--emit-base64", action="store_true", help="Also print LLM_RESULT_PAYLOAD_B64 for legacy copy-paste import.")
    parser.add_argument("--pretty", action="store_true", help="Print the dashboard import JSON envelope.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config_path = resolve_config_path(
            config_path=args.config,
            gemma=args.gemma,
            gemma_variant=args.gemma_variant,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bench_config = load_config(str(config_path) if config_path else None)
    cases = selected_cases(bench_config, args.case)
    dashboard_eligible = bool(bench_config.get("dashboardEligible", True))

    if args.list_cases:
        for case in cases:
            print(
                f"{case['name']}: p={case['promptTokens']} g={case['generationTokens']} "
                f"tier={case.get('tier', 'default')}"
            )
        return 0

    llama_bench_executable = _resolve_llama_bench_executable(args)
    runtime = LlamaCppRuntime(
        executable=llama_bench_executable,
        mock_result_file=args.mock_result_file,
    )
    model = bench_config["model"]
    runtime_config = bench_config["runtime"]
    defaults = bench_config["defaults"]
    model_path = args.model_path or str(resolve_model_path(bench_config))
    device_ids = _resolve_device_ids(args=args, runtime_config=runtime_config)
    host = probe_benchmark_env(requested_backend=args.backend, device_ids=device_ids)
    device = args.device or _resolve_llama_device(runtime_config=runtime_config, host=host, device_ids=device_ids)
    split_mode = args.split_mode or str(runtime_config.get("splitMode", "layer"))
    heterogeneous_devices = _is_heterogeneous_device_set(host)
    results = []

    print("LLM inference benchmark")
    if not dashboard_eligible:
        print("  Mode:        non-dashboard auxiliary run")
    print(f"  Runtime:     {runtime.name}")
    if args.mock_result_file:
        print(f"  llama-bench: mock data from {args.mock_result_file}")
    else:
        print(f"  llama-bench: {runtime.resolve_executable() or 'not found'}")
    print(f"  Model:       {model['displayName']} ({model['artifact']})")
    print(f"  Model path:  {model_path}")
    print(f"  Cases:       {len(cases)}")
    print(
        "  Profile:     "
        f"KV={runtime_config.get('cacheTypeK', 'default')}/{runtime_config.get('cacheTypeV', 'default')}, "
        f"FA={'on' if runtime_config.get('flashAttention') else 'off'}, "
        f"batch={args.batch_size or int(defaults['batchSize'])}, "
        f"ubatch={args.ubatch_size if args.ubatch_size is not None else defaults.get('ubatchSize', 'default')}, "
        f"split={split_mode}, dev={_display_llama_devices(device)}"
    )
    print()

    for index, case in enumerate(cases, start=1):
        context_size = _case_context_size(
            prompt_tokens=int(case["promptTokens"]),
            generation_tokens=int(case["generationTokens"]),
            padding=int(runtime_config.get("contextPadding", 0)),
            rounding=int(runtime_config.get("contextRounding", 1)),
        )
        config = RuntimeConfig(
            case_name=case["name"],
            case_description=case.get("description", ""),
            model=model["key"],
            model_path=model_path,
            base_model=model["baseModel"],
            artifact=model["artifact"],
            prompt_tokens=int(case["promptTokens"]),
            generation_tokens=int(case["generationTokens"]),
            context_size=context_size,
            batch_size=args.batch_size or int(defaults["batchSize"]),
            ubatch_size=args.ubatch_size if args.ubatch_size is not None else defaults.get("ubatchSize"),
            repetitions=args.repetitions or int(defaults["repetitions"]),
            n_gpu_layers=int(runtime_config["nGpuLayers"]),
            device=device,
            device_ids=device_ids,
            split_mode=split_mode,
            heterogeneous_devices=heterogeneous_devices,
            threads=args.threads if args.threads is not None else defaults.get("threads"),
            cache_type_k=runtime_config.get("cacheTypeK"),
            cache_type_v=runtime_config.get("cacheTypeV"),
            flash_attention=bool(runtime_config.get("flashAttention", False)),
        )

        print(
            f"[{index}/{len(cases)}] {config.case_name} "
            f"(p={config.prompt_tokens:,}, g={config.generation_tokens:,}, ctx={config.context_size:,}) ...",
            flush=True,
        )
        try:
            result = runtime.run(config)
        except Exception as exc:
            result = _failed_result(config, str(exc))
        results.append(result)
        _print_result(result)

    _print_summary(results)
    should_build_payload = dashboard_eligible or bool(args.output_file) or bool(args.emit_base64)
    if not should_build_payload:
        print("Non-dashboard run: payload files, debug sidecar, base64 output, and import command skipped.")
        if args.pretty:
            print("--pretty ignored because no dashboard payload was requested.")
        return 0

    if not dashboard_eligible:
        print("Warning: this preset is not dashboard-eligible; generated payload output is for local inspection only.")

    print("Using host metadata from pre-run probe...")
    payload = build_payload(host=host, results=results, note=args.note)
    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    payload_file = None
    debug_file = None
    if dashboard_eligible or args.output_file:
        payload_file = _resolve_output_file(args=args, model_key=str(model["key"]))
        _write_json(payload_file, payload)
        if not args.no_debug_output:
            debug_payload = build_payload(
                host=host,
                results=results,
                note=args.note,
                include_raw_result=True,
            )
            debug_file = _resolve_debug_output_file(args=args, payload_file=payload_file)
            _write_json(debug_file, debug_payload)
        _print_payload_outputs(payload_file=payload_file, debug_file=debug_file, importable=dashboard_eligible)
    if args.emit_base64:
        print(encode_payload(payload))
    return 0


def _parse_device_ids(value: str) -> list[int]:
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [int(chunk) for chunk in chunks] or [0]


def _resolve_llama_bench_executable(args: argparse.Namespace) -> str | None:
    if not args.docker:
        return args.llama_bench
    if args.llama_bench:
        raise SystemExit("--docker and --llama-bench cannot be used together.")
    if args.mock_result_file:
        raise SystemExit("--docker and --mock-result-file cannot be used together.")

    script = Path(__file__).resolve().parents[1] / "scripts" / "run-llama-bench-docker.sh"
    _check_docker_llama_bench_environment(script)
    return str(script)


def _check_docker_llama_bench_environment(script: Path) -> None:
    if not script.is_file():
        raise SystemExit(f"Docker llama-bench runner was not found: {script}")

    print("Checking Docker llama-bench environment...")
    try:
        subprocess.run([str(script), "--check"], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Docker llama-bench environment check failed with exit code {exc.returncode}.") from exc
    print()


def _resolve_device_ids(*, args: argparse.Namespace, runtime_config: dict) -> list[int]:
    if args.gpu_id:
        return _parse_device_ids(args.gpu_id)
    if args.device:
        ids = _device_ids_from_llama_device(args.device)
        if ids:
            return ids
    if bool(runtime_config.get("autoMultiGpu", False)):
        ids = _detect_cuda_device_ids()
        if ids:
            return ids
    return [0]


def _detect_cuda_device_ids() -> list[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    ids = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids


def _device_ids_from_llama_device(value: str) -> list[int]:
    ids = []
    for chunk in value.replace(",", "/").split("/"):
        chunk = chunk.strip()
        if not chunk.upper().startswith("CUDA"):
            continue
        suffix = chunk[4:]
        if suffix.isdigit():
            ids.append(int(suffix))
    return ids


def _resolve_llama_device(*, runtime_config: dict, host: dict, device_ids: list[int]) -> str:
    configured_device = str(runtime_config.get("device", "auto"))
    if host.get("backend") == "cuda" and device_ids:
        return "/".join(f"CUDA{device_id}" for device_id in device_ids)
    return configured_device


def _display_llama_devices(value: str) -> str:
    return value.replace("/", ",")


def _is_heterogeneous_device_set(host: dict) -> bool:
    device_ids = host.get("device_ids")
    device = str(host.get("device", ""))
    if not isinstance(device_ids, list) or len(device_ids) <= 1:
        return False
    repeated_prefix = f"{len(device_ids)}x "
    if not device.startswith(repeated_prefix):
        return True
    return "," in device.removeprefix(repeated_prefix)


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
        contextSize=config.context_size,
        batchSize=config.batch_size,
        ubatchSize=config.ubatch_size,
        repetitions=config.repetitions,
        ppTps=None,
        ppStddev=None,
        tgTps=None,
        tgStddev=None,
        nGpuLayers=config.n_gpu_layers,
        deviceIds=config.device_ids,
        llamaDevices=config.device,
        splitMode=config.split_mode,
        heterogeneousDevices=config.heterogeneous_devices,
        threads=config.threads,
        backendRaw="",
        cacheTypeK=config.cache_type_k or "",
        cacheTypeV=config.cache_type_v or "",
        flashAttention=config.flash_attention,
        modelSizeBytes=None,
        modelParams=None,
        rawResult={},
        note="",
        date="",
    )


def _print_result(result: RuntimeResult) -> None:
    if result.status != "ok":
        print(f"  failed: {result.error or 'unknown error'}")
        print()
        return

    parts = []
    if result.ppTps is not None:
        parts.append(f"PP {_format_rate(result.ppTps)} tok/s")
        if result.ppStddev is not None:
            parts[-1] += f" (stddev {_format_rate(result.ppStddev)})"
    if result.tgTps is not None:
        parts.append(f"TG {_format_rate(result.tgTps)} tok/s")
        if result.tgStddev is not None:
            parts[-1] += f" (stddev {_format_rate(result.tgStddev)})"

    metrics = ", ".join(parts) if parts else "no PP/TG metrics parsed"
    backend = result.accelerationBackend or "unknown backend"
    print(
        f"  ok: {metrics}; backend={backend}; ngl={result.nGpuLayers}; "
        f"ctx={result.contextSize or 'N/A'}; ubatch={result.ubatchSize or 'N/A'}"
    )
    print()


def _print_summary(results: list[RuntimeResult]) -> None:
    ok_count = sum(1 for result in results if result.status == "ok")
    failed_count = len(results) - ok_count
    print("Summary")
    print(f"  Cases:  {len(results)}")
    print(f"  OK:     {ok_count}")
    print(f"  Failed: {failed_count}")
    print()


def _format_rate(value: float) -> str:
    return f"{value:,.2f}"


def _case_context_size(
    *,
    prompt_tokens: int,
    generation_tokens: int,
    padding: int,
    rounding: int,
) -> int:
    needed = prompt_tokens + generation_tokens + max(padding, 0)
    rounding = max(rounding, 1)
    return ((needed + rounding - 1) // rounding) * rounding


def _resolve_output_file(*, args: argparse.Namespace, model_key: str) -> Path:
    if args.output_file:
        return Path(args.output_file).expanduser()

    output_dir = Path(args.output_dir).expanduser()
    timestamp = _payload_timestamp()
    return output_dir / f"llm-bench-{timestamp}-{_safe_filename(model_key)}.json"


def _resolve_debug_output_file(*, args: argparse.Namespace, payload_file: Path) -> Path:
    if args.debug_output_file:
        return Path(args.debug_output_file).expanduser()
    return payload_file.with_name(f"{payload_file.stem}.debug{payload_file.suffix}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _print_payload_outputs(*, payload_file: Path, debug_file: Path | None, importable: bool = True) -> None:
    payload_path = _display_path(payload_file)
    print(f"LLM_RESULT_PAYLOAD_FILE={payload_path}")
    if debug_file is not None:
        print(f"LLM_DEBUG_PAYLOAD_FILE={_display_path(debug_file)}")
    if importable:
        print()
        print("Import:")
        print(f"  python3 scripts/manage-data.py l {shlex.quote(payload_path)}")
    print()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def _payload_timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip("-") or "payload"


if __name__ == "__main__":
    raise SystemExit(main())
