from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from threading import Event
from typing import Any, Callable
from urllib import error, request
from urllib.parse import urlsplit

from llm_bench.api.protocol import make_request, normalize_event, parse_sse, token_usage
from llm_bench.api.estimation import estimate
from llm_bench.api.suite import (
    DECODE_INPUT_CHARS, INPUT_SIZES, OUTPUT_LIMITS, PROBE_OUTPUT_TOKENS,
    REPETITIONS, REQUEST_TIMEOUT, SUITE_ID, cases, make_prompt,
)


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def measure_request(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt_chars: int,
    max_output_tokens: int,
    phase: str,
    timeout: float,
    include_usage: bool,
) -> dict[str, Any]:
    spec = make_request(
        provider,
        base_url,
        model,
        make_prompt(prompt_chars),
        max_output_tokens,
        api_key,
        include_usage,
    )
    sample: dict[str, Any] = {
        "phase": phase,
        "promptChars": prompt_chars,
        "outputLimit": max_output_tokens,
        "status": "ok",
        "error": "",
        "headersMs": None,
        "ttftMs": None,
        "lastContentMs": None,
        "totalMs": None,
        "contentEvents": 0,
        "contentTimeline": [],
        "finishReason": None,
        "outputChars": 0,
        "promptTokens": None,
        "outputTokens": None,
        "reasoningTokens": None,
        "cachedPromptTokens": None,
        "clientDecodeTps": None,
        "serverPrefillTps": None,
        "serverDecodeTps": None,
        "serverQueueMs": None,
    }
    usage: dict[str, Any] = {}
    complete = False
    opener = request.build_opener(NoRedirect)
    started = time.perf_counter()
    try:
        http_request = request.Request(
            spec.url, data=spec.body, headers=spec.headers, method="POST"
        )
        with opener.open(http_request, timeout=timeout) as response:
            sample["headersMs"] = _ms(time.perf_counter() - started)
            for event_name, data in parse_sse(response):
                now = time.perf_counter()
                if now - started > timeout:
                    raise TimeoutError("Request exceeded suite deadline")
                event = normalize_event(provider, event_name, data)
                if event.finish_reason is not None:
                    sample["finishReason"] = event.finish_reason
                if event.usage:
                    usage.update(event.usage)
                if event.text:
                    sample["contentEvents"] += 1
                    sample["outputChars"] += len(event.text)
                    sample["contentTimeline"].append({"timeMs": _ms(now - started), "chars": len(event.text)})
                    if sample["ttftMs"] is None:
                        sample["ttftMs"] = _ms(now - started)
                    sample["lastContentMs"] = _ms(now - started)
                if event.kind == "error":
                    raise ValueError("Provider returned an error event")
                if event.kind == "done":
                    complete = True
                    break
        if not complete:
            raise ValueError("Stream ended without a completion event")
    except error.HTTPError as exc:
        sample["status"], sample["error"] = "failed", f"HTTP {exc.code}"
        exc.close()
    except UnicodeEncodeError:
        sample["status"], sample["error"] = "failed", "Cannot encode request URL or authentication headers; check Base URL and API key"
    except (OSError, ValueError, UnicodeError) as exc:
        sample["status"], sample["error"] = "failed", type(exc).__name__
    finally:
        sample["totalMs"] = _ms(time.perf_counter() - started)

    prompt_tokens, output_tokens, reasoning_tokens = token_usage(provider, usage)
    sample["promptTokens"] = prompt_tokens
    sample["outputTokens"] = output_tokens
    sample["reasoningTokens"] = reasoning_tokens
    sample["cachedPromptTokens"] = _cached_tokens(provider, usage)
    if sample["status"] == "ok" and sample["ttftMs"] is None and not (type(output_tokens) is int and output_tokens > 0):
        sample["status"], sample["error"] = "failed", "No generated tokens or visible text in stream"
    return sample


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _cached_tokens(
    provider: str, usage: dict[str, Any]
) -> int | None:
    if provider == "anthropic":
        return usage.get("cache_read_input_tokens")
    if provider == "gemini":
        return usage.get("cachedContentTokenCount")
    if provider == "openai-responses":
        return (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if "prompt_cache_hit_tokens" in usage:
        return usage["prompt_cache_hit_tokens"]
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_pos = (len(ordered) - 1) * 0.95
    lower = int(p95_pos)
    p95 = ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (p95_pos - lower)
    return {"p50": round(statistics.median(ordered), 3), "p95": round(p95, 3)}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [s for s in samples if s["phase"] != "warmup"]
    success = [s for s in measured if s["status"] == "ok"]
    decode_samples = [s for s in success if s["phase"] == "decode"]
    estimates = estimate(measured)
    cache_counts = [s["cachedPromptTokens"] for s in success]
    if any(count is not None and count > 0 for count in cache_counts):
        cache_status = (
            "mixed" if any(count == 0 for count in cache_counts) else "cached"
        )
    elif cache_counts and all(count == 0 for count in cache_counts):
        cache_status = "uncached"
    else:
        cache_status = "unknown"

    def rates(key: str, group: list[dict[str, Any]]) -> dict[str, float] | None:
        return _distribution([s[key] for s in group if s[key] is not None])

    return {
        "successes": len(success),
        "failures": len(measured) - len(success),
        "cacheStatus": cache_status,
        "ttftMs": rates("ttftMs", decode_samples),
        "totalMs": rates("totalMs", decode_samples),
        "clientDecodeTps": None,
        "serverDecodeTps": None,
        "serverPrefillTps": None,
        "decodeTps": estimates["decode"]["value"],
        "effectivePrefillTps": estimates["prefill"]["value"],
        "prefillFitR2": estimates["prefill"].get("fitR2"),
        "estimates": estimates,
        "promptTokens": rates("promptTokens", decode_samples),
        "outputTokens": rates("outputTokens", decode_samples),
    }


def run_benchmark(
    *, provider: str, base_url: str, model: str, api_key: str | None,
    include_usage: bool, region: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: Event | None = None,
) -> dict[str, Any]:
    work = cases()
    stop = cancel if cancel is not None else Event()
    samples: list[dict[str, Any]] = []
    total = len(work)

    def report(phase: str, sample: dict[str, Any] | None = None) -> None:
        if on_progress is not None:
            on_progress({"completed": len(samples), "total": total, "phase": phase, "sample": sample})
        elif sample is not None:
            print(f"[{len(samples)}/{total}] {phase}: {sample['status']}; TTFT={sample['ttftMs']} ms, total={sample['totalMs']} ms", flush=True)
        else:
            print(f"[{len(samples) + 1}/{total}] {phase}", flush=True)

    for case in work:
        if stop.is_set():
            break
        report(case.phase)
        sample = measure_request(
            provider=provider, base_url=base_url, model=model, api_key=api_key,
            prompt_chars=case.prompt_chars, max_output_tokens=case.output_limit,
            phase=case.phase, timeout=REQUEST_TIMEOUT, include_usage=include_usage,
        )
        samples.append(sample)
        report(case.phase, sample)
        if case.phase == "warmup" and sample["status"] != "ok":
            raise RuntimeError(f"Warmup failed: {sample['error']}")
    summary = summarize(samples)
    summary["requestThroughputRps"] = None
    return {
        "schemaVersion": "2.0", "kind": "llm-api",
        "runStatus": "cancelled" if stop.is_set() and len(samples) < total else "completed",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": {
            "suiteId": SUITE_ID, "provider": provider, "model": model,
            "endpointHost": urlsplit(base_url).hostname,
            "promptChars": list(INPUT_SIZES), "repetitions": REPETITIONS,
            "streamUsageRequested": include_usage,
            "probeOutputTokens": PROBE_OUTPUT_TOKENS,
            "decodeOutputTokens": max(OUTPUT_LIMITS),
            "decodeOutputLimits": list(OUTPUT_LIMITS),
            "decodePromptChars": DECODE_INPUT_CHARS,
            "timeoutSeconds": REQUEST_TIMEOUT, "concurrency": 1, "clientRegion": region,
        },
        "summary": summary, "samples": samples,
    }
