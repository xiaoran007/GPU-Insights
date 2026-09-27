from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from llm_bench.api.benchmark import run_benchmark
from llm_bench.api.credentials import KEY_VARIABLES, resolve_api_key
from llm_bench.api.protocol import DEFAULT_URLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure LLM API latency and token throughput.")
    parser.add_argument("--provider", required=True, choices=sorted(KEY_VARIABLES))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="API base URL, including /v1 where required.")
    parser.add_argument("--api-key", help="API key. Prefer an environment variable or .env to avoid shell history.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--no-stream-usage", action="store_true",
                        help="Omit stream_options for compatible servers that reject it.")
    parser.add_argument("--prompt-chars", default="512,2048,8192", help="Comma-separated input sizes for prefill estimation.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--probe-output-tokens", type=int, default=32)
    parser.add_argument("--decode-output-tokens", type=int, default=256)
    parser.add_argument("--decode-prompt-chars", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--max-prompt-chars", type=int, default=100000)
    parser.add_argument("--max-output-tokens", type=int, default=10000,
                        help="Maximum aggregate requested output tokens for this run.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm-api-bench"))
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Show the workload without making API requests.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sizes = [int(value.strip()) for value in args.prompt_chars.split(",")]
    except ValueError as exc:
        raise SystemExit("--prompt-chars must contain comma-separated integers") from exc
    if len(set(sizes)) < 3 or any(size <= 0 for size in sizes):
        raise SystemExit("--prompt-chars needs at least three distinct positive sizes")
    if args.repetitions < 2 or min(args.probe_output_tokens, args.decode_output_tokens,
                                   args.decode_prompt_chars) <= 0 or args.timeout <= 0:
        raise SystemExit("Repetitions must be >= 2; output limits, decode prompt size, and timeout must be positive")
    request_count = 1 + args.repetitions * (len(sizes) + 1)
    total_prompt_chars = args.decode_prompt_chars * (args.repetitions + 1) + args.repetitions * sum(sizes)
    total_output_tokens = args.probe_output_tokens * (1 + args.repetitions * len(sizes)) + args.decode_output_tokens * args.repetitions
    if (request_count > args.max_requests or total_prompt_chars > args.max_prompt_chars
            or total_output_tokens > args.max_output_tokens):
        raise SystemExit("Planned workload exceeds a configured request, input, or output limit")
    if args.provider == "compatible" and not args.base_url:
        raise SystemExit("--base-url is required for --provider compatible")
    base_url = args.base_url or DEFAULT_URLS[args.provider]
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise SystemExit("--base-url must start with http:// or https://")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise SystemExit("--base-url cannot contain credentials, query parameters, or fragments")
    print(f"Provider: {args.provider}; model: {args.model}; endpoint: {parsed_url.netloc}")
    print(f"Planned requests: {request_count}; prompt characters: {total_prompt_chars}; maximum output tokens: {total_output_tokens}")
    if args.dry_run:
        return 0
    api_key = resolve_api_key(args.provider, args.api_key, args.env_file)
    if args.provider not in {"compatible", "vllm", "llama-cpp"} and not api_key:
        raise SystemExit(f"Missing key: use --api-key, {KEY_VARIABLES[args.provider]}, or {args.env_file}")
    payload = run_benchmark(provider=args.provider, base_url=base_url, model=args.model,
                            api_key=api_key, prompt_chars=sizes, repetitions=args.repetitions,
                            probe_output_tokens=args.probe_output_tokens,
                            decode_output_tokens=args.decode_output_tokens,
                            decode_prompt_chars=args.decode_prompt_chars, timeout=args.timeout,
                            include_usage=not args.no_stream_usage)
    output = args.output_file or args.output_dir / f"llm-api-{payload['createdAt'][:19].replace(':', '-')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Result: {output}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["summary"]["successes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
