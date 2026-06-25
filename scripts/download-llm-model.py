#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_bench.config import load_config, resolve_config_path, resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the fixed GPU-Insights LLM benchmark model.")
    parser.add_argument("--config", help="Path to LLM benchmark config JSON.")
    parser.add_argument("--gemma", action="store_true", help="Use a non-dashboard Gemma small-GPU preset. Requires --12b or --e2b.")
    gemma_size = parser.add_mutually_exclusive_group()
    gemma_size.add_argument("--12b", dest="gemma_variant", action="store_const", const="12b", help="Use the Gemma 4 12B QAT UD-Q4_K_XL preset.")
    gemma_size.add_argument("--e2b", dest="gemma_variant", action="store_const", const="e2b", help="Use the Gemma 4 E2B QAT UD-Q4_K_XL preset.")
    parser.add_argument("--output", help="Override output path. Defaults to the configured localPath.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved download plan without downloading.")
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
    config = load_config(str(config_path) if config_path else None)
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
        print(f"  Expected: {_format_bytes(expected_bytes)} ({expected_bytes:,} bytes)")

    if args.dry_run:
        print(f"  URL:      {url}")
        return 0

    if output_path.exists() and not args.force:
        existing_size = output_path.stat().st_size
        if expected_bytes and existing_size == expected_bytes:
            print("Existing file matches expected size; nothing to do.")
            return 0
        if expected_bytes:
            print(
                "Existing file size does not match expected size: "
                f"{_format_bytes(existing_size)} ({existing_size:,} bytes) vs "
                f"{_format_bytes(expected_bytes)} ({expected_bytes:,} bytes)."
            )
        else:
            print(f"Existing file found: {_format_bytes(existing_size)} ({existing_size:,} bytes).")
        print("Pass --force to overwrite.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    if args.force and temp_path.exists():
        temp_path.unlink()

    resume_from = 0
    if temp_path.exists():
        resume_from = temp_path.stat().st_size
        if expected_bytes and resume_from == expected_bytes:
            temp_path.replace(output_path)
            print("Existing partial file matches expected size; promoted it to final output.")
            return 0
        if expected_bytes and resume_from > expected_bytes:
            print(
                "Existing partial file is larger than expected: "
                f"{_format_bytes(resume_from)} ({resume_from:,} bytes) vs "
                f"{_format_bytes(expected_bytes)} ({expected_bytes:,} bytes)."
            )
            print("Remove the .part file or pass --force to restart.")
            return 1
        if resume_from > 0:
            print(f"Resuming from {_format_bytes(resume_from)} ({resume_from:,} bytes).")

    request = urllib.request.Request(url)
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        if resume_from > 0 and status != 206:
            print("Server did not honor the Range request; restarting from 0 bytes.")
            temp_path.unlink()
            resume_from = 0

        content_length = int(response.headers.get("Content-Length") or 0)
        if expected_bytes:
            total = expected_bytes
        elif resume_from > 0 and status == 206:
            total = resume_from + content_length
        else:
            total = content_length

        downloaded = resume_from
        mode = "ab" if resume_from > 0 and status == 206 else "wb"
        progress = ProgressReporter(downloaded)
        progress.update(downloaded, total, force=True)
        with temp_path.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                progress.update(downloaded, total)
        progress.update(downloaded, total, force=True)
    print()

    final_size = temp_path.stat().st_size
    if expected_bytes and final_size != expected_bytes:
        print(
            "Downloaded size mismatch: "
            f"got {_format_bytes(final_size)} ({final_size:,} bytes), "
            f"expected {_format_bytes(expected_bytes)} ({expected_bytes:,} bytes)."
        )
        print(f"Partial file kept at {temp_path} for resume/debugging.")
        return 1

    temp_path.replace(output_path)

    print(f"Downloaded model to {output_path}")
    return 0


class ProgressReporter:
    def __init__(self, initial_downloaded: int) -> None:
        self.initial_downloaded = initial_downloaded
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0

    def update(self, downloaded: int, total: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_rendered_at < 0.2:
            return

        self.last_rendered_at = now
        speed = self._bytes_per_second(downloaded, now)
        if total > 0:
            pct = min(downloaded / total, 1.0)
            print(
                "\r  Downloaded: "
                f"{_progress_bar(pct)} "
                f"{pct * 100:5.1f}% "
                f"{_format_bytes(downloaded)} / {_format_bytes(total)} "
                f"@ {_format_rate(speed)}",
                end="",
                flush=True,
            )
        else:
            print(
                "\r  Downloaded: "
                f"{_format_bytes(downloaded)} "
                f"@ {_format_rate(speed)}",
                end="",
                flush=True,
            )

    def _bytes_per_second(self, downloaded: int, now: float) -> float:
        elapsed = max(now - self.started_at, 1e-6)
        return max(downloaded - self.initial_downloaded, 0) / elapsed


def _progress_bar(progress: float, width: int = 28) -> str:
    filled = min(width, max(0, int(round(progress * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_rate(bytes_per_second: float) -> str:
    return f"{_format_bytes(int(bytes_per_second))}/s"


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024


if __name__ == "__main__":
    raise SystemExit(main())
