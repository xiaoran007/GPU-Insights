"""The versioned, bounded API workload shared by every endpoint."""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass


SUITE_ID = "api-standard-v2"
REPETITIONS = 3
INPUT_SIZES = (512, 2048, 8192, 16384)
OUTPUT_LIMITS = (128, 256, 512)
DECODE_INPUT_CHARS = 1024
PROBE_OUTPUT_TOKENS = 32
REQUEST_TIMEOUT = 120


@dataclass(frozen=True)
class Case:
    phase: str
    prompt_chars: int
    output_limit: int


def cases() -> list[Case]:
    warmups = [Case("warmup", DECODE_INPUT_CHARS, 64)] * 2
    # Interleave every length in each block to reduce time/order confounding.
    measured = []
    rng = random.Random(0)
    for _ in range(REPETITIONS):
        block = [Case("prefill", size, PROBE_OUTPUT_TOKENS) for size in INPUT_SIZES]
        block += [Case("decode", DECODE_INPUT_CHARS, limit) for limit in OUTPUT_LIMITS]
        rng.shuffle(block)
        measured.extend(block)
    return warmups + measured


def plan() -> dict[str, int]:
    work = cases()
    return {
        "requests": len(work),
        "promptChars": sum(case.prompt_chars for case in work),
        "outputTokens": sum(case.output_limit for case in work),
    }


def description() -> dict:
    return {
        "id": SUITE_ID, "warmups": 2, "repetitions": REPETITIONS,
        "inputSizes": list(INPUT_SIZES), "outputLimits": list(OUTPUT_LIMITS),
        "decodeInputChars": DECODE_INPUT_CHARS, **plan(),
    }


def make_prompt(char_count: int) -> str:
    # Diversify the earliest user tokens on every request, including warmups.
    # A suffix nonce would leave the preceding prompt reusable by prefix caches.
    prefix = (
        f"{uuid.uuid4().hex}\n"
        "Task: Write a catalog of 200 numbered fictional laboratory records. "
        "For each record give a name, material, purpose, and handling procedure. "
        "Use a separate full sentence for every field. Start with record 1. "
        "Continue producing records without an introduction, summary, or closing remark.\n"
        "Reference vocabulary:\n"
    )
    unit = "glass copper ceramic sensor sample vessel bracket valve filter thermal optical magnetic. "
    remaining = char_count - len(prefix)
    if remaining < 0:
        raise ValueError("Prompt size is smaller than the suite instruction")
    return prefix + (unit * (remaining // len(unit) + 1))[:remaining]
