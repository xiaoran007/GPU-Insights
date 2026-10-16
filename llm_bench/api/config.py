"""Shared configuration and workload validation for CLI and local UI clients."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

from llm_bench.api.credentials import KEY_VARIABLES
from llm_bench.api.protocol import DEFAULT_URLS


@dataclass(frozen=True)
class BenchmarkConfig:
    provider: str = "compatible"
    model: str = ""
    base_url: str = ""
    region: str = ""
    prompt_chars: list[int] = field(default_factory=lambda: [512, 2048, 8192])
    repetitions: int = 3
    concurrency: int = 1
    probe_output_tokens: int = 32
    decode_output_tokens: int = 256
    decode_prompt_chars: int = 2048
    timeout: float = 120
    max_requests: int = 30
    max_prompt_chars: int = 100000
    max_output_tokens: int = 10000
    include_usage: bool = True
    allow_insecure_http: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BenchmarkConfig:
        if not isinstance(values, dict) or values.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown benchmark configuration fields")
        return cls(**values).validated()

    def validated(self) -> BenchmarkConfig:
        for name in ("provider", "model", "base_url", "region"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError(f"Invalid {name}")
        if self.provider not in KEY_VARIABLES:
            raise ValueError("Unknown provider")
        if not self.model.strip():
            raise ValueError("Model is required")
        if (
            not isinstance(self.prompt_chars, list)
            or not self.prompt_chars
            or any(type(size) is not int or size <= 0 for size in self.prompt_chars)
            or len(set(self.prompt_chars)) < 3
        ):
            raise ValueError("Prompt sizes need at least three distinct positive integers")
        for name in (
            "repetitions", "concurrency", "probe_output_tokens", "decode_output_tokens",
            "decode_prompt_chars", "max_requests", "max_prompt_chars", "max_output_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.repetitions < 2:
            raise ValueError("Repetitions must be >= 2")
        if (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Timeout must be a finite positive number")
        if type(self.include_usage) is not bool or type(self.allow_insecure_http) is not bool:
            raise ValueError("Usage and insecure HTTP options must be booleans")
        plan = self.plan()
        if (
            plan["requests"] > self.max_requests
            or plan["promptChars"] > self.max_prompt_chars
            or plan["outputTokens"] > self.max_output_tokens
        ):
            raise ValueError("Planned workload exceeds a configured request, input, or output limit")
        base_url = self.base_url.strip() or DEFAULT_URLS.get(self.provider, "")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Base URL must start with http:// or https://")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Base URL cannot contain credentials, query parameters, or fragments")
        # Accessing port also validates malformed port specifications.
        parsed.port
        return replace(self, model=self.model.strip(), base_url=base_url.rstrip("/"), region=self.region.strip())

    def plan(self) -> dict[str, int]:
        if self.concurrency == 1:
            return {
                "requests": 1 + self.repetitions * (len(self.prompt_chars) + 1),
                "promptChars": self.decode_prompt_chars * (self.repetitions + 1)
                + self.repetitions * sum(self.prompt_chars),
                "outputTokens": self.probe_output_tokens * (1 + self.repetitions * len(self.prompt_chars))
                + self.decode_output_tokens * self.repetitions,
            }
        count = self.repetitions * self.concurrency
        return {
            "requests": 1 + count,
            "promptChars": self.decode_prompt_chars * (1 + count),
            "outputTokens": self.probe_output_tokens + self.decode_output_tokens * count,
        }

    def validate_key(self, api_key: str | None) -> None:
        if self.provider not in {"compatible", "vllm", "llama-cpp"} and not api_key:
            raise ValueError(f"Missing key: configure {KEY_VARIABLES[self.provider]} or enter a session key")
        parsed = urlsplit(self.base_url)
        if (
            api_key and parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            and not self.allow_insecure_http
        ):
            raise ValueError("Refusing to send an API key over non-loopback HTTP; use HTTPS or explicitly allow insecure HTTP")

    def run_options(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("max_requests", "max_prompt_chars", "max_output_tokens", "allow_insecure_http"):
            values.pop(name)
        return values
