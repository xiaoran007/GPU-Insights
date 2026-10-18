"""Connection configuration; the benchmark workload is fixed by its suite."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any
from urllib.parse import urlsplit

from llm_bench.api.credentials import KEY_VARIABLES
from llm_bench.api.protocol import DEFAULT_URLS
from llm_bench.api.suite import plan


@dataclass(frozen=True)
class BenchmarkConfig:
    provider: str = "compatible"
    model: str = ""
    base_url: str = ""
    region: str = ""
    include_usage: bool = True
    allow_insecure_http: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BenchmarkConfig:
        if not isinstance(values, dict) or values.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("Only connection settings are accepted; the standard suite fixes the workload")
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
        if type(self.include_usage) is not bool or type(self.allow_insecure_http) is not bool:
            raise ValueError("Usage and insecure HTTP options must be booleans")
        base_url = self.base_url.strip() or DEFAULT_URLS.get(self.provider, "")
        if not base_url.isascii():
            raise ValueError("Base URL contains non-ASCII characters; use the server's ASCII or URL-encoded address")
        if any(ord(char) <= 32 or ord(char) == 127 for char in base_url):
            raise ValueError("Base URL cannot contain spaces or control characters")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Base URL must start with http:// or https://")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Base URL cannot contain credentials, query parameters, or fragments")
        parsed.port
        return replace(self, model=self.model.strip(), base_url=base_url.rstrip("/"), region=self.region.strip())

    @staticmethod
    def plan() -> dict[str, int]:
        return plan()

    def validate_key(self, api_key: str | None) -> None:
        if api_key and any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("API key contains unsupported characters; check for spaces, line breaks, non-ASCII characters, or smart quotes")
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
        values.pop("allow_insecure_http")
        return values
