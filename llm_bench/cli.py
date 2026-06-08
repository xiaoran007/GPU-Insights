from __future__ import annotations

import argparse
import json

from llm_bench.payload import build_payload, encode_payload
from llm_bench.runtimes.base import RuntimeConfig
from llm_bench.runtimes.llama_cpp import LlamaCppRuntime
from scripts.probe_benchmark_env import probe_benchmark_env


DEFAULT_MODEL_KEY = "qwen3_6_27b_q4"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_ARTIFACT = "GGUF Q4_K_M"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone LLM inference benchmark launcher.")
    parser.add_argument("--runtime", choices=["llama.cpp"], default="llama.cpp")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, help="Dashboard model key.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model file.")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable.")
    parser.add_argument("--mock-result-file", help="Read llama-bench JSON output from this file.")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=2048)
    parser.add_argument("-n", "--generation-tokens", type=int, default=128)
    parser.add_argument("-b", "--batch-size", type=int, default=2048)
    parser.add_argument("-r", "--repetitions", type=int, default=5)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=-1)
    parser.add_argument("-dev", "--device", default="auto")
    parser.add_argument("-t", "--threads", type=int)
    parser.add_argument("--backend", default="auto", help="Host metadata backend probe.")
    parser.add_argument("--gpu-id", default="0", help="Comma-separated device ids for metadata probing.")
    parser.add_argument("--note", default="")
    parser.add_argument("--pretty", action="store_true", help="Print decoded JSON before the payload line.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = RuntimeConfig(
        model=args.model,
        model_path=args.model_path,
        base_model=args.base_model,
        artifact=args.artifact,
        prompt_tokens=args.prompt_tokens,
        generation_tokens=args.generation_tokens,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
        n_gpu_layers=args.n_gpu_layers,
        device=args.device,
        threads=args.threads,
    )

    runtime = LlamaCppRuntime(
        executable=args.llama_bench,
        mock_result_file=args.mock_result_file,
    )
    result = runtime.run(config)
    host = probe_benchmark_env(requested_backend=args.backend, device_ids=_parse_device_ids(args.gpu_id))
    payload = build_payload(host=host, results=[result], note=args.note)

    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(encode_payload(payload))
    return 0


def _parse_device_ids(value: str) -> list[int]:
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [int(chunk) for chunk in chunks] or [0]


if __name__ == "__main__":
    raise SystemExit(main())
