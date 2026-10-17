"""Endpoint-independent length/latency estimates with separate confidence evidence.

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
    return {"value": None, "status": status, "confidence": "unavailable", "reason": reason, **evidence}


def _warn(result: dict[str, Any], reason: str) -> None:
    result["reason"] = "; ".join(filter(None, [result["reason"], reason]))
    if result["value"] is not None:
        result.update(status="unstable", confidence="low")


def _nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _eligible(sample: dict[str, Any]) -> bool:
    return (
        sample["status"] == "ok"
        and _nonnegative(sample.get("promptTokens"))
        and _nonnegative(sample.get("outputTokens"))
        and _nonnegative(sample.get("totalMs")) and sample["totalMs"] > 0
    )


def _fit(points: list[tuple[int, float, float]], groups_required: int, min_span: int) -> dict[str, Any]:
    groups: dict[int, list[tuple[float, float]]] = {}
    for group, x, y in points:
        groups.setdefault(group, []).append((x, y))
    evidence = {"samples": len(points), "groups": len(groups)}
    if len(groups) < 2:
        return _result("insufficient", "Need at least two observed token lengths to calculate a slope", **evidence)
    centers = sorted((statistics.median(x for x, _ in group), statistics.median(y for _, y in group)) for group in groups.values())
    span = centers[-1][0] - centers[0][0]
    evidence["tokenSpan"] = round(span, 3)
    if span <= 0:
        return _result("insufficient", "Observed token lengths are identical; a slope cannot be calculated", **evidence)
    reasons = []
    limited = len(groups) < groups_required or any(len(group) < MIN_REPEATS for group in groups.values())
    if limited:
        reasons.append("Limited sample coverage: fewer than three usable requests at every suite length")
    if span < min_span or len({x for x, _ in centers}) < groups_required:
        limited = True
        reasons.append("Observed token lengths cover a limited range")
    center_slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in combinations(centers, 2) if x2 - x1 >= span / 4]
    slope = statistics.median(center_slopes)
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
    evidence.update(slopeMsPerToken=slope, interceptMs=round(intercept, 3), fitR2=round(r2, 4))
    if slope <= 0:
        return _result("unstable", "Fitted latency slope is non-positive; positive token throughput is undefined", **evidence)
    spread = (_quantile(slopes, 0.75) - _quantile(slopes, 0.25)) / (2 * slope)
    evidence["relativeSlopeSpread"] = round(spread, 4)
    if r2 < MIN_R2:
        reasons.append("Length/latency centers deviate from a linear relationship")
    if spread > MAX_SLOPE_SPREAD:
        reasons.append("Repeated measurements give dispersed slopes")
    # These heuristics label evidence; they never suppress a calculable rate and
    # are not calibrated probabilities that the estimate is accurate.
    return {
        "value": round(1000 / slope, 3),
        "status": "insufficient" if limited else "unstable" if reasons else "ok",
        "confidence": "low" if reasons else "high", "reason": "; ".join(reasons), **evidence,
    }


def estimate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    decode = [s for s in samples if s["phase"] == "decode" and _eligible(s)]
    prompts = [s["promptTokens"] for s in decode]
    output = _fit([(s["outputLimit"], s["outputTokens"], s["totalMs"]) for s in decode], 3, 128)
    if prompts and max(prompts) - min(prompts) > max(16, statistics.median(prompts) * 0.05):
        _warn(output, "Decode requests did not have comparable input token counts")
    if any((s.get("cachedPromptTokens") or 0) > 0 for s in decode):
        _warn(output, "Known prompt-cache hits: rate describes the observed cached workload")
    inputs = [s for s in samples if s["phase"] == "prefill" and _eligible(s)]
    if inputs and len({s["outputTokens"] for s in inputs}) == 1:
        # A constant output cost is part of the intercept, so it requires no
        # decode slope to identify the incremental input cost.
        prefill = _fit([(s["promptChars"], s["promptTokens"], s["totalMs"]) for s in inputs], 4, 512)
        prefill["outputCostAdjustment"] = "constant output count absorbed by intercept"
    elif output["value"] is not None:
        prefill = _fit([
            (s["promptChars"], s["promptTokens"], s["totalMs"] - output["slopeMsPerToken"] * s["outputTokens"])
            for s in inputs
        ], 4, 512)
        if output["confidence"] == "low" and prefill["value"] is not None:
            _warn(prefill, "Output-cost adjustment uses a low-confidence decode estimate")
    else:
        prefill = _result("insufficient", "Variable output counts require a calculable output-cost slope", samples=len(inputs))
    if any((s.get("cachedPromptTokens") or 0) > 0 for s in inputs):
        _warn(prefill, "Known prompt-cache hits: rate describes the observed cached workload")
    # Slopes are retained for auditability, never calibrated against a backend.
    return {"decode": output, "prefill": prefill}
