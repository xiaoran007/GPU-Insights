from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "ttftMs",
    "totalMs",
    "clientDecodeTps",
    "serverDecodeTps",
    "serverPrefillTps",
    "promptTokens",
    "outputTokens",
)


def dashboard_entry(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") != "llm-api" or payload.get("schemaVersion") != "1.0":
        raise ValueError("Expected an llm-api version 1.0 payload")
    config, summary = payload.get("config"), payload.get("summary")
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("Payload must contain config and summary objects")
    for key in ("provider", "model", "endpointHost"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Invalid config.{key}")
    for key in ("successes", "failures"):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise ValueError(f"Invalid summary.{key}")
    if not summary["successes"]:
        raise ValueError("Cannot publish a run without successful requests")
    for key in METRICS:
        value = summary.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Invalid summary.{key}")
        for percentile in ("p50", "p95"):
            item = value.get(percentile)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item < 0
            ):
                raise ValueError(f"Invalid summary.{key}.{percentile}")
    for key in ("effectivePrefillTps", "prefillFitR2", "requestThroughputRps"):
        value = summary.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid summary.{key}")
    if (
        summary.get("effectivePrefillTps") is not None
        and summary["effectivePrefillTps"] <= 0
    ):
        raise ValueError("Invalid summary.effectivePrefillTps")
    if (
        summary.get("requestThroughputRps") is not None
        and summary["requestThroughputRps"] < 0
    ):
        raise ValueError("Invalid summary.requestThroughputRps")
    if not isinstance(payload.get("createdAt"), str) or not payload["createdAt"]:
        raise ValueError("Invalid createdAt")
    if not isinstance(config.get("promptChars"), list) or not all(
        type(size) is int and size > 0 for size in config["promptChars"]
    ):
        raise ValueError("Invalid config.promptChars")
    for key in ("repetitions", "decodeOutputTokens"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Invalid config.{key}")
    if type(config.get("concurrency")) is not int or config["concurrency"] <= 0:
        raise ValueError("Invalid config.concurrency")
    if type(config.get("streamUsageRequested")) is not bool:
        raise ValueError("Invalid config.streamUsageRequested")
    if not isinstance(config.get("clientRegion"), str):
        raise ValueError("Invalid config.clientRegion")
    if summary.get("cacheStatus") not in {"unknown", "uncached", "cached", "mixed"}:
        raise ValueError("Invalid summary.cacheStatus")
    if not isinstance(payload.get("samples"), list):
        raise ValueError("Payload must contain a samples array")
    identity = {
        "createdAt": payload.get("createdAt"),
        "config": config,
        "summary": summary,
        "samples": payload["samples"],
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "runId": run_id,
        "createdAt": payload["createdAt"],
        "provider": config["provider"],
        "model": config["model"],
        "endpointHost": config["endpointHost"],
        "clientRegion": config["clientRegion"],
        "promptChars": config["promptChars"],
        "repetitions": config["repetitions"],
        "concurrency": config["concurrency"],
        "decodeOutputTokens": config["decodeOutputTokens"],
        "streamUsageRequested": config["streamUsageRequested"],
        "summary": {
            key: summary.get(key)
            for key in (
                "successes",
                "failures",
                "cacheStatus",
                *METRICS,
                "effectivePrefillTps",
                "prefillFitR2",
                "requestThroughputRps",
            )
        },
    }


def import_payload(payload_file: Path, data_file: Path, dry_run: bool = False) -> str:
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    entry = dashboard_entry(payload)
    data = json.loads(data_file.read_text(encoding="utf-8"))
    entries = data["benchmarks"]
    if any(old["runId"] == entry["runId"] for old in entries):
        return "Duplicate run; no changes"
    if not dry_run:
        entries.append(entry)
        data["metadata"]["lastUpdated"] = datetime.now(timezone.utc).date().isoformat()
        data_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return f"{'Validated' if dry_run else 'Imported'} {entry['runId']} ({entry['provider']} / {entry['model']})"
