from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from llm_bench.api.config import BenchmarkConfig

from llm_bench.api.benchmark import run_benchmark
from llm_bench.api.credentials import KEY_VARIABLES, resolve_api_key
from llm_bench.api.suite import SUITE_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure LLM API latency and token throughput."
    )
    parser.add_argument("--provider", required=True, choices=sorted(KEY_VARIABLES))
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--region", default="", help="Client location label for dashboard comparisons."
    )
    parser.add_argument(
        "--base-url", help="API base URL, including /v1 where required."
    )
    parser.add_argument(
        "--api-key",
        help="API key. Prefer an environment variable or .env to avoid shell history.",
    )
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="Allow sending an API key to a non-loopback HTTP endpoint.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--no-stream-usage",
        action="store_true",
        help="Omit stream_options for compatible servers that reject it.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/llm-api-bench")
    )
    parser.add_argument("--output-file", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the workload without making API requests.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = BenchmarkConfig(
            provider=args.provider, model=args.model, base_url=args.base_url or "",
            region=args.region, include_usage=not args.no_stream_usage,
            allow_insecure_http=args.allow_insecure_http,
        ).validated()
        plan = config.plan()
        print(f"Provider: {config.provider}; model: {config.model}; endpoint: {urlsplit(config.base_url).netloc}")
        print(
            f"Suite: {SUITE_ID}; planned requests: {plan['requests']}; "
            f"prompt characters: {plan['promptChars']}; maximum output tokens: {plan['outputTokens']}"
        )
        if args.dry_run:
            return 0
        api_key = resolve_api_key(config.provider, args.api_key, args.env_file)
        config.validate_key(api_key)
        payload = run_benchmark(**config.run_options(), api_key=api_key)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    output = (
        args.output_file
        or args.output_dir
        / f"llm-api-{payload['createdAt'][:19].replace(':', '-')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Result: {output}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["summary"]["successes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
