import json
import unittest
from unittest.mock import patch

from llm_bench.api.benchmark import measure_request
from llm_bench.api.protocol import normalize_event


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "hidden reasoning"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            {"choices": [], "timings": {"tokens_cached": 999, "predicted_ms": 0.001}, "usage": {"prompt_tokens": 100, "completion_tokens": 128,
                                       "completion_tokens_details": {"reasoning_tokens": 128}}},
        ]
        for event in events:
            yield f"data: {json.dumps(event)}\n".encode()
            yield b"\n"
        yield b"data: [DONE]\n"
        yield b"\n"


class ProtocolTests(unittest.TestCase):
    def test_output_cap_is_a_completed_response_with_usage(self):
        event = normalize_event("openai-responses", "", json.dumps({
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "max_output_tokens"},
                         "usage": {"input_tokens": 100, "output_tokens": 128}},
        }))
        self.assertEqual(event.kind, "done")
        self.assertEqual(event.finish_reason, "length")
        self.assertEqual(event.usage["output_tokens"], 128)
        bad = normalize_event("openai-responses", "", json.dumps({
            "type": "response.incomplete", "response": {"incomplete_details": {"reason": "content_filter"}},
        }))
        self.assertEqual(bad.kind, "error")

    @patch("llm_bench.api.benchmark.request.build_opener")
    def test_reasoning_only_completion_keeps_latency_and_counts_without_fabricating_ttft(self, opener):
        opener.return_value.open.return_value = Response()
        sample = measure_request(
            provider="compatible", base_url="http://localhost/v1", model="test", api_key=None,
            prompt_chars=1024, max_output_tokens=128, phase="decode", timeout=120, include_usage=True,
        )
        self.assertEqual(sample["status"], "ok")
        self.assertEqual(sample["outputTokens"], 128)
        self.assertEqual(sample["reasoningTokens"], 128)
        self.assertEqual(sample["finishReason"], "length")
        self.assertIsNone(sample["ttftMs"])
        self.assertEqual(sample["contentTimeline"], [])
        self.assertIsNone(sample["clientDecodeTps"])
        self.assertIsNone(sample["serverDecodeTps"])
        self.assertIsNone(sample["cachedPromptTokens"])


if __name__ == "__main__":
    unittest.main()
