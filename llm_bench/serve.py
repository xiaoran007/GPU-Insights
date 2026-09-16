from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_bench.config import ROOT_DIR, load_config, resolve_config_path, resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a configured GGUF model with llama-server.")
    parser.add_argument("--config", help="Path to LLM model config JSON.")
    parser.add_argument("--qwen38", action="store_true", help="Use Qwen3.8-27B UD-Q6_K.")
    parser.add_argument("--gemma", action="store_true", help="Use Gemma with --12b or --e2b.")
    gemma_size = parser.add_mutually_exclusive_group()
    gemma_size.add_argument("--12b", dest="gemma_variant", action="store_const", const="12b")
    gemma_size.add_argument("--e2b", dest="gemma_variant", action="store_const", const="e2b")
    parser.add_argument("--llama-server", help="Path to the source-built llama-server binary.")
    parser.add_argument("--model-path", help="Override the configured GGUF path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--ctx-size", type=int, help="Total context tokens across all slots; must divide evenly by --parallel.")
    context.add_argument("--ctx-per-slot", type=int, help="Context tokens per request, including output. Defaults to the serving preset.")
    parser.add_argument("--parallel", type=int, help="Concurrent request slots. Defaults to the serving preset.")
    parser.add_argument("--cache-type-k", help="Override the serving preset's K cache type.")
    parser.add_argument("--cache-type-v", help="Override the serving preset's V cache type.")
    parser.add_argument("--device", help="llama.cpp device selection, e.g. CUDA0.")
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="Extra llama-server arguments after --.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    try:
        config_path = resolve_config_path(
            config_path=args.config,
            qwen38=args.qwen38,
            gemma=args.gemma,
            gemma_variant=args.gemma_variant,
        )
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config(str(config_path) if config_path else None)
    serving = config.get("serving", {})
    parallel = args.parallel if args.parallel is not None else serving.get("parallel", 1)
    context_per_slot = args.ctx_per_slot if args.ctx_per_slot is not None else serving.get("contextPerSlot", 32768)
    if parallel <= 0 or context_per_slot <= 0:
        parser.error("--parallel and --ctx-per-slot must be positive.")
    context_size = args.ctx_size if args.ctx_size is not None else context_per_slot * parallel
    if context_size <= 0 or context_size % parallel:
        parser.error("--ctx-size must be positive and divide evenly by --parallel.")
    context_per_slot = context_size // parallel
    if context_per_slot > config["model"]["contextLength"]:
        parser.error("Per-slot context exceeds the configured model contextLength.")
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else resolve_model_path(config)
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}. Download the selected model with scripts/download-llm-model.py first.")

    source_dir = Path(os.environ.get("GPU_INSIGHTS_LLAMA_CPP_DIR", ROOT_DIR / "third_party/llama.cpp")).expanduser()
    build_dir = Path(os.environ.get("GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR", source_dir / "build")).expanduser()
    executable = Path(args.llama_server).expanduser() if args.llama_server else build_dir / "bin/llama-server"
    executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(
            f"llama-server is not executable: {executable}. "
            "Run bash scripts/bootstrap-llama-cpp.sh --backend cuda --prebuilt off "
            "(select your backend), or pass --llama-server."
        )

    runtime = config["runtime"]
    command = [
        str(executable), "--model", str(model_path),
        "--alias", config["model"]["key"],
        "--host", args.host, "--port", str(args.port),
        "--ctx-size", str(context_size), "--parallel", str(parallel),
        "--no-kv-unified", "--cont-batching",
        "--n-gpu-layers", str(runtime["nGpuLayers"]),
        "--split-mode", runtime["splitMode"],
        "--cache-type-k", args.cache_type_k or serving.get("cacheTypeK", runtime["cacheTypeK"]),
        "--cache-type-v", args.cache_type_v or serving.get("cacheTypeV", runtime["cacheTypeV"]),
        "--flash-attn", "on" if runtime["flashAttention"] else "off",
        "--jinja",
    ]
    device = args.device or runtime.get("device")
    if device and device != "auto":
        command.extend(["--device", device])
    extra_args = args.server_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    command.extend(extra_args)

    print(f"Starting llama-server: {config['model']['displayName']}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Context: {context_per_slot:,} tokens/slot x {parallel} slots = {context_size:,} total", flush=True)
    print(f"Listen: {args.host}:{args.port}", flush=True)
    # Replace the launcher so the server receives termination signals directly.
    os.execv(str(executable), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
