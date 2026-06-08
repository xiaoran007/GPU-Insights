from __future__ import annotations

import base64
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

from llm_bench.runtimes.base import RuntimeResult


PAYLOAD_PREFIX = "LLM_RESULT_PAYLOAD_B64="
PAYLOAD_SCHEMA_VERSION = "1.0"


def current_date_string() -> str:
    now = datetime.now()
    return f"{now.year}.{now.month}.{now.day}"


def build_payload(
    *,
    host: Dict[str, Any],
    results: List[RuntimeResult],
    note: str,
) -> Dict[str, Any]:
    return {
        "schemaVersion": PAYLOAD_SCHEMA_VERSION,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "host": host,
        "benchmarks": [
            {
                **asdict(result),
                "date": result.date or current_date_string(),
                "note": result.note or note,
            }
            for result in results
        ],
    }


def encode_payload(payload: Dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return f"{PAYLOAD_PREFIX}{encoded.decode('ascii')}"

