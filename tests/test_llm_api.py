import threading
import unittest
from unittest.mock import patch

from llm_bench.api.benchmark import run_benchmark
from llm_bench.api.config import BenchmarkConfig
from llm_bench.api.data import dashboard_entry


def sample(**kwargs):
    return {
        "phase": kwargs["phase"], "promptChars": kwargs["prompt_chars"],
        "status": "ok", "error": "", "ttftMs": 10, "totalMs": 100,
        "cachedPromptTokens": 0, "promptTokens": 20, "outputTokens": 10,
        "clientDecodeTps": 90, "serverDecodeTps": None, "serverPrefillTps": None,
    }


class ConfigurationTests(unittest.TestCase):
    def test_default_plan_includes_warmup(self):
        self.assertEqual(BenchmarkConfig().plan(), {
            "requests": 13, "promptChars": 40448, "outputTokens": 1088,
        })

    def test_invalid_and_unsafe_configuration(self):
        for values in ({"timeout": float("nan")}, {"concurrency": True},
                       {"prompt_chars": [1, 1, 2]}, {"max_requests": 1},
                       {"base_url": "https://user:secret@example.org/v1"},
                       {"base_url": "https://example.org/?key=secret"},
                       {"unexpected": "value"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BenchmarkConfig.from_dict({"model": "test", "base_url": "http://localhost:8000/v1", **values})
        config = BenchmarkConfig(provider="compatible", model="test", base_url="http://example.org/v1").validated()
        with self.assertRaises(ValueError):
            config.validate_key("secret")

    @patch("llm_bench.api.benchmark.measure_request", side_effect=sample)
    def test_cancel_preserves_completed_samples_and_payload_compatibility(self, measure):
        stop = threading.Event()
        events = []

        def progress(event):
            events.append(event)
            if event["completed"] == 2:
                stop.set()

        config = BenchmarkConfig(model="test", base_url="http://localhost:8000/v1").validated()
        payload = run_benchmark(**config.run_options(), api_key="secret", cancel=stop, on_progress=progress)
        self.assertEqual(measure.call_count, 2)
        self.assertEqual(payload["runStatus"], "cancelled")
        self.assertEqual(payload["summary"]["successes"], 1)
        self.assertEqual(dashboard_entry(payload)["model"], "test")
        self.assertNotIn("secret", str(payload))
        self.assertEqual(events[-1]["completed"], 2)

    @patch("llm_bench.api.benchmark.measure_request", side_effect=sample)
    def test_concurrent_cancellation_stops_new_work(self, measure):
        stop = threading.Event()

        def progress(event):
            if event["completed"] > 1:
                stop.set()

        config = BenchmarkConfig(model="test", base_url="http://localhost/v1", concurrency=2).validated()
        result = run_benchmark(**config.run_options(), api_key=None, cancel=stop, on_progress=progress)
        self.assertLessEqual(measure.call_count, 3)
        self.assertEqual(result["runStatus"], "cancelled")
        self.assertIsNone(result["summary"]["effectivePrefillTps"])


if __name__ == "__main__":
    unittest.main()
