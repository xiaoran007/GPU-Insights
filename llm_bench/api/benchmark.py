from __future__ import annotations

import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit

from llm_bench.api.protocol import make_request, normalize_event, parse_sse, token_usage


PROMPT_UNIT = "Analyze the following project note and keep the answer concise. A benchmark must report measured latency, request conditions, and uncertainty. "


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def make_prompt(char_count: int) -> str:
    prefix = f"Run identifier: {uuid.uuid4().hex}.\n"
    body = (PROMPT_UNIT * ((char_count // len(PROMPT_UNIT)) + 1))[:char_count]
    return (
        prefix
        + body
        + "\nGive a numbered list of practical observations. Continue until the output limit."
    )


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
    timings: dict[str, Any] = {}
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
                event = normalize_event(provider, event_name, data)
                if event.usage:
                    usage.update(event.usage)
                if event.timings:
                    timings.update(event.timings)
                if event.text:
                    sample["contentEvents"] += 1
                    sample["outputChars"] += len(event.text)
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
    except (OSError, ValueError, UnicodeError) as exc:
        sample["status"], sample["error"] = "failed", type(exc).__name__
    finally:
        sample["totalMs"] = _ms(time.perf_counter() - started)

    prompt_tokens, output_tokens, reasoning_tokens = token_usage(provider, usage)
    sample["promptTokens"] = prompt_tokens
    sample["outputTokens"] = output_tokens
    sample["reasoningTokens"] = reasoning_tokens
    sample["cachedPromptTokens"] = _cached_tokens(provider, usage, timings)
    sample["serverPrefillTps"], sample["serverDecodeTps"], sample["serverQueueMs"] = (
        _server_rates(timings, output_tokens)
    )
    if sample["status"] == "ok" and sample["ttftMs"] is None:
        sample["status"], sample["error"] = "failed", "No visible text in stream"
    if sample["status"] == "ok" and output_tokens is not None:
        visible_tokens = output_tokens - (reasoning_tokens or 0)
        window_ms = sample["lastContentMs"] - sample["ttftMs"]
        if visible_tokens >= 2 and sample["contentEvents"] >= 2 and window_ms > 0:
            sample["clientDecodeTps"] = round(
                (visible_tokens - 1) * 1000 / window_ms, 3
            )
    return sample


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _cached_tokens(
    provider: str, usage: dict[str, Any], timings: dict[str, Any]
) -> int | None:
    if provider == "anthropic":
        return usage.get("cache_read_input_tokens")
    if provider == "gemini":
        return usage.get("cachedContentTokenCount")
    if provider == "openai-responses":
        return (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if "prompt_cache_hit_tokens" in usage:
        return usage["prompt_cache_hit_tokens"]
    if "tokens_cached" in timings:
        return timings["tokens_cached"]
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def _server_rates(
    timings: dict[str, Any], output_tokens: int | None
) -> tuple[float | None, float | None, float | None]:
    if "prompt_ms" in timings:
        prompt_ms, predicted_ms = timings.get("prompt_ms"), timings.get("predicted_ms")
        pp = (
            (timings.get("prompt_n") or 0) * 1000 / prompt_ms
            if prompt_ms and prompt_ms > 0
            else None
        )
        tg = (
            (timings.get("predicted_n") or 0) * 1000 / predicted_ms
            if predicted_ms and predicted_ms > 0
            else None
        )
        return _round_rate(pp), _round_rate(tg), None
    generation_ms = timings.get("generation_time_ms")
    tg = (
        (output_tokens - 1) * 1000 / generation_ms
        if output_tokens and output_tokens > 1 and generation_ms and generation_ms > 0
        else None
    )
    return None, _round_rate(tg), timings.get("queue_time_ms")


def _round_rate(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


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


def _prefill_regression(
    samples: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    groups: dict[int, list[tuple[int, float]]] = {}
    for sample in samples:
        if (
            sample["phase"] != "prefill"
            or sample["status"] != "ok"
            or sample["promptTokens"] is None
            or sample["cachedPromptTokens"] not in (None, 0)
        ):
            continue
        groups.setdefault(sample["promptChars"], []).append(
            (sample["promptTokens"], sample["ttftMs"])
        )
    points = [
        (statistics.median(x for x, _ in group), statistics.median(y for _, y in group))
        for group in groups.values()
        if len(group) >= 2
    ]
    if len(points) < 3:
        return None, None
    xbar = statistics.mean(x for x, _ in points)
    ybar = statistics.mean(y for _, y in points)
    x_variance = sum((x - xbar) ** 2 for x, _ in points)
    if x_variance == 0:
        return None, None
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / x_variance
    total = sum((y - ybar) ** 2 for _, y in points)
    residual = sum((y - (ybar + slope * (x - xbar))) ** 2 for x, y in points)
    r2 = 1 - residual / total if total > 0 else 0
    if slope <= 0 or r2 < 0.7:
        return None, round(r2, 3)
    return round(1000 / slope, 3), round(r2, 3)


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [s for s in samples if s["phase"] != "warmup"]
    success = [s for s in measured if s["status"] == "ok"]
    decode_samples = [s for s in success if s["phase"] == "decode"]
    prefill_samples = [s for s in success if s["phase"] == "prefill"]
    prefill_rate, fit = _prefill_regression(success)
    cache_counts = [s["cachedPromptTokens"] for s in decode_samples]
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
        "clientDecodeTps": rates("clientDecodeTps", decode_samples),
        "serverDecodeTps": rates("serverDecodeTps", decode_samples),
        "serverPrefillTps": rates("serverPrefillTps", prefill_samples),
        "effectivePrefillTps": prefill_rate,
        "prefillFitR2": fit,
        "promptTokens": rates("promptTokens", decode_samples),
        "outputTokens": rates("outputTokens", decode_samples),
    }


def run_benchmark(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt_chars: list[int],
    repetitions: int,
    probe_output_tokens: int,
    decode_output_tokens: int,
    decode_prompt_chars: int,
    timeout: float,
    include_usage: bool,
    concurrency: int,
    region: str,
) -> dict[str, Any]:
    if concurrency == 1:
        work = [
            ("prefill", size, probe_output_tokens)
            for size in prompt_chars
            for _ in range(repetitions)
        ]
        work += [
            ("decode", decode_prompt_chars, decode_output_tokens)
            for _ in range(repetitions)
        ]
        random.Random(0).shuffle(work)
    else:
        work = [("decode", decode_prompt_chars, decode_output_tokens)] * (
            concurrency * repetitions
        )
    print(f"[1/{len(work) + 1}] Warmup", flush=True)
    samples = [
        measure_request(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt_chars=decode_prompt_chars,
            max_output_tokens=probe_output_tokens,
            phase="warmup",
            timeout=timeout,
            include_usage=include_usage,
        )
    ]
    if samples[0]["status"] != "ok":
        raise RuntimeError(f"Warmup failed: {samples[0]['error']}")
    started = time.perf_counter()
    if concurrency == 1:
        for index, (phase, size, limit) in enumerate(work, 2):
            print(
                f"[{index}/{len(work) + 1}] {phase}: {size} prompt characters, {limit} output tokens",
                flush=True,
            )
            sample = measure_request(
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt_chars=size,
                max_output_tokens=limit,
                phase=phase,
                timeout=timeout,
                include_usage=include_usage,
            )
            samples.append(sample)
            print(
                f"  {sample['status']}: TTFT={sample['ttftMs']} ms, total={sample['totalMs']} ms",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    measure_request,
                    provider=provider,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    prompt_chars=size,
                    max_output_tokens=limit,
                    phase=phase,
                    timeout=timeout,
                    include_usage=include_usage,
                )
                for phase, size, limit in work
            ]
            for index, future in enumerate(as_completed(futures), 2):
                sample = future.result()
                samples.append(sample)
                print(
                    f"[{index}/{len(work) + 1}] {sample['status']}: TTFT={sample['ttftMs']} ms, total={sample['totalMs']} ms",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    summary = summarize(samples)
    summary["requestThroughputRps"] = (
        round(summary["successes"] / elapsed, 3)
        if concurrency > 1 and elapsed > 0
        else None
    )
    return {
        "schemaVersion": "1.0",
        "kind": "llm-api",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": {
            "provider": provider,
            "model": model,
            "endpointHost": urlsplit(base_url).hostname,
            "promptChars": prompt_chars,
            "repetitions": repetitions,
            "streamUsageRequested": include_usage,
            "probeOutputTokens": probe_output_tokens,
            "decodeOutputTokens": decode_output_tokens,
            "decodePromptChars": decode_prompt_chars,
            "timeoutSeconds": timeout,
            "concurrency": concurrency,
            "clientRegion": region,
        },
        "summary": summary,
        "samples": samples,
    }
