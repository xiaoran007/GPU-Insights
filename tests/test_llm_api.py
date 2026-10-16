import copy
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
        "outputLimit": kwargs["max_output_tokens"],
        "clientDecodeTps": None, "serverDecodeTps": None, "serverPrefillTps": None,
    }


class ConfigurationTests(unittest.TestCase):
    def test_default_plan_includes_warmup(self):
        self.assertEqual(BenchmarkConfig().plan(), {
            "requests": 23, "promptChars": 92672, "outputTokens": 3200,
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
            if event["completed"] == 3:
                stop.set()

        config = BenchmarkConfig(model="test", base_url="http://localhost:8000/v1").validated()
        payload = run_benchmark(**config.run_options(), api_key="secret", cancel=stop, on_progress=progress)
        self.assertEqual(measure.call_count, 3)
        self.assertEqual(payload["runStatus"], "cancelled")
        self.assertEqual(payload["summary"]["successes"], 1)
        with self.assertRaisesRegex(ValueError, "incomplete run"):
            dashboard_entry(payload)
        self.assertNotIn("secret", str(payload))
        self.assertEqual(events[-1]["completed"], 3)

    @patch("llm_bench.api.benchmark.measure_request", side_effect=sample)
    def test_v2_collection_preserves_evidence_and_rejects_inconsistent_rates(self, measure):
        config = BenchmarkConfig(model="test", base_url="http://localhost/v1").validated()
        payload = run_benchmark(**config.run_options(), api_key=None, on_progress=lambda _: None)
        entry = dashboard_entry(payload)
        self.assertEqual(entry["suiteId"], "api-standard-v2")
        self.assertEqual(entry["summary"]["estimates"]["decode"]["status"], "insufficient")
        changed = copy.deepcopy(payload)
        changed["summary"]["decodeTps"] = 100
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            dashboard_entry(changed)
        legacy = copy.deepcopy(payload)
        legacy["schemaVersion"] = "1.0"
        legacy["config"].pop("suiteId")
        self.assertNotIn("suiteId", dashboard_entry(legacy))

    def test_suite_controls_are_not_user_configuration(self):
        for field in ("concurrency", "repetitions", "timeout", "prompt_chars"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                BenchmarkConfig.from_dict({"model": "test", "base_url": "http://localhost/v1", field: 2})


if __name__ == "__main__":
    unittest.main()
