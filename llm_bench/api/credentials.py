from __future__ import annotations

import os
from pathlib import Path


KEY_VARIABLES = {
    "openai-chat": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "compatible": "LLM_API_KEY",
    "vllm": "LLM_API_KEY",
    "llama-cpp": "LLM_API_KEY",
}


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env assignment at line {line_number}")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not name.isidentifier():
            raise ValueError(f"Invalid .env variable name at line {line_number}")
        if value.startswith(('"', "'")):
            quote = value[0]
            if not value.endswith(quote) or len(value) < 2:
                raise ValueError(f"Unclosed .env quote at line {line_number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def resolve_api_key(provider: str, cli_key: str | None, env_file: Path) -> str | None:
    if cli_key is not None:
        return cli_key
    variable = KEY_VARIABLES[provider]
    return os.environ.get(variable) or read_env_file(env_file).get(variable)
