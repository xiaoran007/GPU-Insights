"""Endpoint-independent length/latency estimates with explicit evidence gates.

No chunk is assumed to contain one token. Only complete-request latency and
reported token counts enter the fits; server timing extensions never enter them.
"""
from __future__ import annotations

import math
import statistics
from itertools import combinations
from typing import Any


MIN_REPEATS = 3
MAX_SLOPE_SPREAD = 0.25
MIN_R2 = 0.95


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _result(status: str, reason: str, **evidence: Any) -> dict[str, Any]:
    return {"value": None, "status": status, "reason": reason, **evidence}


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _eligible(sample: dict[str, Any]) -> bool:
    return (
        sample["status"] == "ok"
        and _positive(sample.get("promptTokens"))
        and _positive(sample.get("outputTokens"))
        and _positive(sample.get("totalMs"))
        and sample.get("cachedPromptTokens") in (None, 0)
    )


def _fit(points: list[tuple[int, float, float]], groups_required: int, min_span: int) -> dict[str, Any]:
    groups: dict[int, list[tuple[float, float]]] = {}
    for group, x, y in points:
        groups.setdefault(group, []).append((x, y))
    evidence = {"samples": len(points), "groups": len(groups)}
    if len(groups) < groups_required or any(len(group) < MIN_REPEATS for group in groups.values()):
        return _result("insufficient", "Need three usable requests at every suite length", **evidence)
    centers = sorted((statistics.median(x for x, _ in group), statistics.median(y for _, y in group)) for group in groups.values())
    span = centers[-1][0] - centers[0][0]
    evidence["tokenSpan"] = round(span, 3)
    if span < min_span or len({x for x, _ in centers}) < groups_required:
        return _result("insufficient", "Actual token lengths do not cover a sufficient range", **evidence)
    center_slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in combinations(centers, 2) if x2 - x1 >= span / 4]
    slope = statistics.median(center_slopes)
    if slope <= 0:
        return _result("unstable", "Latency does not increase consistently with token length", **evidence)
    intercept = statistics.median(y - slope * x for x, y in centers)
    mean = statistics.mean(y for _, y in centers)
    variation = sum((y - mean) ** 2 for _, y in centers)
    residual = sum((y - intercept - slope * x) ** 2 for x, y in centers)
    r2 = 1 - residual / variation if variation else 0
    # Cross-length repeat pairs expose jitter hidden by fitting group medians.
    # Their IQR tolerates isolated outliers, consistent with the median estimator.
    slopes = []
    for (g1, x1, y1), (g2, x2, y2) in combinations(points, 2):
        if g1 != g2 and abs(x2 - x1) >= span / 4:
            slopes.append((y2 - y1) / (x2 - x1))
    spread = (_quantile(slopes, 0.75) - _quantile(slopes, 0.25)) / (2 * slope)
    evidence.update(fitR2=round(r2, 4), relativeSlopeSpread=round(spread, 4))
    if r2 < MIN_R2 or spread > MAX_SLOPE_SPREAD:
        return _result("unstable", "Length/latency relationship is noisy or nonlinear", **evidence)
    return {
        "value": round(1000 / slope, 3), "status": "ok", "reason": "",
        "slopeMsPerToken": slope, "interceptMs": round(intercept, 3), **evidence,
    }


def estimate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    decode = [s for s in samples if s["phase"] == "decode" and _eligible(s) and s["outputTokens"] >= 64]
    prompts = [s["promptTokens"] for s in decode]
    if prompts and max(prompts) - min(prompts) > max(16, statistics.median(prompts) * 0.05):
        output = _result("unstable", "Decode requests did not have comparable input token counts", samples=len(decode))
    else:
        output = _fit([(s["outputLimit"], s["outputTokens"], s["totalMs"]) for s in decode], 3, 128)
    if output["status"] != "ok":
        prefill = _result("insufficient", "Input estimate requires a reliable output-cost estimate", samples=0)
    else:
        prefill = _fit([
            (s["promptChars"], s["promptTokens"], s["totalMs"] - output["slopeMsPerToken"] * s["outputTokens"])
            for s in samples if s["phase"] == "prefill" and _eligible(s)
        ], 4, 512)
    # Slopes are retained for auditability, never calibrated against a backend.
    return {"decode": output, "prefill": prefill}
