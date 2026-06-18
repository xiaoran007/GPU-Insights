#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_bench.config import load_config, resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the fixed GPU-Insights LLM benchmark model.")
    parser.add_argument("--config", help="Path to LLM benchmark config JSON.")
    parser.add_argument("--output", help="Override output path. Defaults to the configured localPath.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved download plan without downloading.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    model = config["model"]
    output_path = Path(args.output).expanduser() if args.output else resolve_model_path(config)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    url = (
        f"https://huggingface.co/{model['hfRepo']}/resolve/"
        f"{model['hfRevision']}/{model['filename']}"
    )
    expected_bytes = model.get("expectedBytes")

    print("LLM benchmark model download")
    print(f"  Repo:     {model['hfRepo']}")
    print(f"  Revision: {model['hfRevision']}")
    print(f"  File:     {model['filename']}")
    print(f"  Output:   {output_path}")
    if expected_bytes:
        print(f"  Expected: {expected_bytes:,} bytes")

    if args.dry_run:
        print(f"  URL:      {url}")
        return 0

    if output_path.exists() and not args.force:
        if expected_bytes and output_path.stat().st_size == expected_bytes:
            print("Existing file matches expected size; nothing to do.")
            return 0
        print("Existing file found. Pass --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or expected_bytes or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, total)
        print()
        if expected_bytes and temp_path.stat().st_size != expected_bytes:
            print(
                f"Downloaded size mismatch: got {temp_path.stat().st_size:,}, "
                f"expected {expected_bytes:,}."
            )
            return 1
        temp_path.replace(output_path)
    finally:
        if temp_path.exists() and not output_path.exists():
            temp_path.unlink()

    print(f"Downloaded model to {output_path}")
    return 0


def _print_progress(downloaded: int, total: int) -> None:
    if total > 0:
        pct = downloaded / total * 100
        print(f"\r  Downloaded: {downloaded:,}/{total:,} bytes ({pct:5.1f}%)", end="", flush=True)
    else:
        print(f"\r  Downloaded: {downloaded:,} bytes", end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
