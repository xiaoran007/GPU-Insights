"""Generic timing models: no inference engine, tokenizer, or server metrics required."""
import copy
import random
import unittest

from llm_bench.api.estimation import estimate
from llm_bench.api.suite import cases, make_prompt, plan


def synthetic_samples(*, chunk_size=1, latency=250, prefill_ms=0.5, decode_ms=10, jitter=0):
    rng = random.Random(99)
    samples = []
    for case in cases():
        prompt_tokens = case.prompt_chars // 4
        output_tokens = case.output_limit
        first_chunk_tokens = min(chunk_size, output_tokens)
        first = latency + prompt_tokens * prefill_ms + first_chunk_tokens * decode_ms
        total = latency + prompt_tokens * prefill_ms + output_tokens * decode_ms + rng.uniform(-jitter, jitter)
        samples.append({
            "phase": case.phase, "status": "ok", "promptChars": case.prompt_chars,
            "outputLimit": output_tokens, "promptTokens": prompt_tokens,
            "outputTokens": output_tokens, "cachedPromptTokens": 0,
            "ttftMs": first, "lastContentMs": total, "totalMs": total,
            "contentEvents": (output_tokens + chunk_size - 1) // chunk_size,
            "finishReason": "length",
        })
    return samples


class EstimationTests(unittest.TestCase):
    def test_fixed_suite_budget_and_exact_prompt_sizes(self):
        self.assertEqual(plan(), {"requests": 23, "promptChars": 92672, "outputTokens": 3200})
        for case in cases():
            self.assertEqual(len(make_prompt(case.prompt_chars)), case.prompt_chars)
        measured = cases()[2:]
        for block in (measured[:7], measured[7:14], measured[14:]):
            self.assertEqual(len({(case.phase, case.prompt_chars, case.output_limit) for case in block}), 7)

    def test_chunk_partition_and_fixed_overhead_do_not_change_rates(self):
        for chunk_size in (1, 10, 64, 10000):
            for overhead in (0, 250, 4000):
                with self.subTest(chunk_size=chunk_size, overhead=overhead):
                    result = estimate(synthetic_samples(chunk_size=chunk_size, latency=overhead))
                    self.assertEqual(result["decode"]["status"], "ok")
                    self.assertAlmostEqual(result["decode"]["value"], 100, places=3)
                    self.assertAlmostEqual(result["prefill"]["value"], 2000, places=3)

    def test_rates_generalize_across_speeds_and_moderate_jitter(self):
        for prefill_ms, decode_ms in ((0.1, 2), (0.5, 10), (2, 40)):
            with self.subTest(prefill_ms=prefill_ms, decode_ms=decode_ms):
                result = estimate(synthetic_samples(prefill_ms=prefill_ms, decode_ms=decode_ms, jitter=15))
                for name, reference in (("decode", 1000 / decode_ms), ("prefill", 1000 / prefill_ms)):
                    self.assertEqual(result[name]["status"], "ok")
                    self.assertLess(abs(result[name]["value"] / reference - 1), 0.05)

    def test_isolated_delays_do_not_overrule_repeated_consistent_measurements(self):
        for phase in ("decode", "prefill"):
            samples = synthetic_samples(jitter=5)
            next(sample for sample in samples if sample["phase"] == phase)["totalMs"] += 1000
            result = estimate(samples)
            for metric, reference in (("decode", 100), ("prefill", 2000)):
                with self.subTest(delayed_phase=phase, metric=metric):
                    self.assertEqual(result[metric]["status"], "ok")
                    self.assertLess(abs(result[metric]["value"] / reference - 1), 0.03)

    def test_original_short_output_failure_is_not_reported_as_throughput(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample.update(outputTokens=12, contentEvents=2, totalMs=450, ttftMs=405)
        result = estimate(samples)
        self.assertIsNone(result["decode"]["value"])
        self.assertIsNone(result["prefill"]["value"])

    def test_missing_usage_or_known_cache_cannot_supply_a_fit(self):
        for field, value in (("outputTokens", None), ("promptTokens", None), ("cachedPromptTokens", 100)):
            samples = synthetic_samples()
            for sample in samples:
                sample[field] = value
            with self.subTest(field=field):
                self.assertIsNone(estimate(samples)["decode"]["value"])

    def test_matched_input_is_required_for_decode(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample["promptTokens"] = sample["outputLimit"] * 4
        self.assertEqual(estimate(samples)["decode"]["status"], "unstable")

    def test_unstable_and_nonlinear_services_are_not_given_precise_rates(self):
        for mode in ("noise", "nonlinear"):
            samples = synthetic_samples(jitter=2200 if mode == "noise" else 0)
            if mode == "nonlinear":
                for sample in samples:
                    if sample["phase"] == "decode":
                        sample["totalMs"] += sample["outputTokens"] ** 3 / 10000
            with self.subTest(mode=mode):
                self.assertEqual(estimate(samples)["decode"]["status"], "unstable")

    def test_warmups_server_extensions_and_content_events_are_not_estimator_inputs(self):
        samples = synthetic_samples()
        expected = estimate(samples)
        changed = copy.deepcopy(samples)
        for sample in changed:
            sample.update(serverDecodeTps=99999999, serverPrefillTps=1, contentEvents=1, ttftMs=99999)
            if sample["phase"] == "warmup":
                sample.update(totalMs=99999999, outputTokens=99999)
        self.assertEqual(estimate(changed), expected)

    def test_failed_repetitions_do_not_silently_change_the_suite_population(self):
        samples = synthetic_samples()
        next(sample for sample in samples if sample["phase"] == "decode")["status"] = "failed"
        self.assertEqual(estimate(samples)["decode"]["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
