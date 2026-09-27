from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote


DEFAULT_URLS = {
    "openai-chat": "https://api.openai.com/v1",
    "openai-responses": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "deepseek": "https://api.deepseek.com",
    "vllm": "http://127.0.0.1:8000/v1",
    "llama-cpp": "http://127.0.0.1:8080/v1",
}


@dataclass(frozen=True)
class RequestSpec:
    url: str
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None


def make_request(
    provider: str, base_url: str, model: str, prompt: str,
    max_output_tokens: int, api_key: str | None, include_usage: bool,
) -> RequestSpec:
    base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if provider == "anthropic":
        url = f"{base}/messages"
        headers.update({"x-api-key": api_key or "", "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": max_output_tokens, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
    elif provider == "gemini":
        url = f"{base}/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
        headers["x-goog-api-key"] = api_key or ""
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_output_tokens}}
    elif provider == "openai-responses":
        url = f"{base}/responses"
        headers["Authorization"] = f"Bearer {api_key}"
        body = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens,
                "stream": True}
    else:
        url = f"{base}/chat/completions"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": True}
        body["max_completion_tokens" if provider == "openai-chat" else "max_tokens"] = max_output_tokens
        if include_usage:
            body["stream_options"] = {"include_usage": True}
    return RequestSpec(url, headers, json.dumps(body, ensure_ascii=False).encode("utf-8"))


def parse_sse(lines: Iterator[bytes]) -> Iterator[tuple[str, str]]:
    event_name = ""
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event_name, "\n".join(data)
            event_name, data = "", []
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event_name, "\n".join(data)


def normalize_event(provider: str, event_name: str, raw_data: str) -> StreamEvent:
    if raw_data == "[DONE]":
        return StreamEvent("done")
    data = json.loads(raw_data)
    if provider == "anthropic":
        kind = data.get("type", event_name)
        if kind == "error":
            return StreamEvent("error")
        if kind == "message_stop":
            return StreamEvent("done")
        if kind == "content_block_delta":
            delta = data.get("delta") or {}
            return StreamEvent("content", text=delta.get("text", "") if delta.get("type") == "text_delta" else "")
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            return StreamEvent("content", text=block.get("text", "") if block.get("type") == "text" else "")
        if kind == "message_start":
            return StreamEvent("usage", usage=(data.get("message") or {}).get("usage"))
        if kind == "message_delta":
            return StreamEvent("usage", usage=data.get("usage"))
    elif provider == "gemini":
        if "error" in data:
            return StreamEvent("error")
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        finished = any(candidate.get("finishReason") for candidate in data.get("candidates") or [])
        return StreamEvent("done" if finished else "content", text="".join(part.get("text", "") for part in parts),
                           usage=data.get("usageMetadata"))
    elif provider == "openai-responses":
        kind = data.get("type", event_name)
        if kind in {"error", "response.failed", "response.incomplete"}:
            return StreamEvent("error")
        if kind == "response.output_text.delta":
            return StreamEvent("content", text=data.get("delta", ""))
        if kind == "response.completed":
            response = data.get("response") or {}
            return StreamEvent("done", usage=response.get("usage"), timings=response.get("metrics"))
    else:
        if "error" in data:
            return StreamEvent("error")
        choices = data.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        return StreamEvent("content", text=delta.get("content") or "",
                           usage=data.get("usage"),
                           timings=data.get("timings") or data.get("metrics"))
    return StreamEvent("other")


def token_usage(provider: str, usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if not usage:
        return None, None, None
    if provider == "anthropic":
        prompt = sum(int(usage.get(key) or 0) for key in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return (prompt or None, usage.get("output_tokens"),
                (usage.get("output_tokens_details") or {}).get("thinking_tokens"))
    if provider == "gemini":
        return usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount")
    if provider == "openai-responses":
        return (usage.get("input_tokens"), usage.get("output_tokens"),
                (usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
    return (usage.get("prompt_tokens"), usage.get("completion_tokens"),
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
