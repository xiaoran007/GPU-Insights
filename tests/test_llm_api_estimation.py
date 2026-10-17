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
                    self.assertEqual(result["decode"]["confidence"], "high")
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

    def test_identical_output_counts_cannot_identify_decode_but_allow_prefill(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample.update(outputTokens=12, contentEvents=2, totalMs=450, ttftMs=405)
        result = estimate(samples)
        self.assertIsNone(result["decode"]["value"])
        self.assertEqual(result["decode"]["confidence"], "unavailable")
        self.assertEqual(result["prefill"]["value"], 2000)
        self.assertEqual(result["prefill"]["confidence"], "high")

    def test_short_or_narrow_lengths_still_produce_an_estimate(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample["outputTokens"] = sample["outputLimit"] // 16
                sample["totalMs"] = 250 + sample["promptTokens"] * 0.5 + sample["outputTokens"] * 10
        output = estimate(samples)["decode"]
        self.assertEqual(output["value"], 100)
        self.assertEqual(output["confidence"], "low")
        self.assertIn("limited range", output["reason"])

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
        output = estimate(samples)["decode"]
        self.assertEqual(output["status"], "unstable")
        self.assertEqual(output["confidence"], "low")
        self.assertEqual(output["value"], 100)

    def test_noise_and_nonlinearity_lower_confidence_without_suppressing_rates(self):
        for mode in ("noise", "nonlinear"):
            samples = synthetic_samples(jitter=2200 if mode == "noise" else 0)
            if mode == "nonlinear":
                for sample in samples:
                    if sample["phase"] == "decode":
                        sample["totalMs"] += sample["outputTokens"] ** 3 / 10000
            with self.subTest(mode=mode):
                output = estimate(samples)["decode"]
                self.assertEqual(output["status"], "unstable")
                self.assertEqual(output["confidence"], "low")
                self.assertGreater(output["value"], 0)
                self.assertIn("fitR2", output)
                self.assertIn("relativeSlopeSpread", output)

    def test_decode_uncertainty_propagates_only_when_input_fit_needs_it(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample["promptTokens"] = sample["outputLimit"] * 4
        self.assertEqual(estimate(samples)["prefill"]["confidence"], "high")
        for sample in samples:
            if sample["phase"] == "prefill":
                sample["outputTokens"] = sample["promptChars"] // 512 + 32
                sample["totalMs"] = 250 + sample["promptTokens"] * 0.5 + sample["outputTokens"] * 10
        prefill = estimate(samples)["prefill"]
        self.assertEqual(prefill["value"], 2000)
        self.assertEqual(prefill["confidence"], "low")
        self.assertIn("low-confidence decode", prefill["reason"])

    def test_high_r2_with_noisy_repeats_preserves_prefill_rate(self):
        samples = synthetic_samples()
        observed = {
            512: [(137, 299.065), (136, 336.647), (137, 305.293)],
            2048: [(363, 327.222), (362, 386.621), (363, 334.176)],
            8192: [(1259, 421.568), (1253, 416.265), (1258, 556.401)],
            16384: [(2455, 673.520), (2456, 573.236), (2451, 552.099)],
        }
        for sample in samples:
            if sample["phase"] == "prefill":
                sample["promptTokens"], sample["totalMs"] = observed[sample["promptChars"]].pop()
        prefill = estimate(samples)["prefill"]
        self.assertEqual(prefill["confidence"], "low")
        self.assertGreater(prefill["value"], 0)
        self.assertGreater(prefill["fitR2"], 0.99)
        self.assertGreater(prefill["relativeSlopeSpread"], 0.25)
        self.assertIn("dispersed slopes", prefill["reason"])

    def test_unidentifiable_cost_retains_diagnostics_without_inventing_a_rate(self):
        for slope in (0, -1):
            samples = synthetic_samples()
            for sample in samples:
                if sample["phase"] == "decode":
                    sample["totalMs"] = 1000 + slope * sample["outputTokens"]
            with self.subTest(slope=slope):
                output = estimate(samples)["decode"]
                self.assertIsNone(output["value"])
                self.assertEqual(output["confidence"], "unavailable")
                self.assertEqual(output["slopeMsPerToken"], slope)
                self.assertIn("fitR2", output)

    def test_variable_outputs_require_an_identifiable_decode_cost(self):
        samples = synthetic_samples()
        for sample in samples:
            if sample["phase"] == "decode":
                sample["outputTokens"] = 12
            elif sample["phase"] == "prefill":
                sample["outputTokens"] += sample["promptChars"] // 512
        self.assertIsNone(estimate(samples)["prefill"]["value"])

    def test_warmups_server_extensions_and_content_events_are_not_estimator_inputs(self):
        samples = synthetic_samples()
        expected = estimate(samples)
        changed = copy.deepcopy(samples)
        for sample in changed:
            sample.update(serverDecodeTps=99999999, serverPrefillTps=1, contentEvents=1, ttftMs=99999)
            if sample["phase"] == "warmup":
                sample.update(totalMs=99999999, outputTokens=99999)
        self.assertEqual(estimate(changed), expected)

    def test_failed_repetitions_lower_confidence_without_suppressing_rates(self):
        samples = synthetic_samples()
        next(sample for sample in samples if sample["phase"] == "decode")["status"] = "failed"
        output = estimate(samples)["decode"]
        self.assertEqual(output["status"], "insufficient")
        self.assertEqual(output["confidence"], "low")
        self.assertEqual(output["value"], 100)
        self.assertEqual(output["samples"], 8)

    def test_two_lengths_with_one_request_each_still_identify_a_slope(self):
        samples = [sample for sample in synthetic_samples() if sample["phase"] == "decode"]
        samples = [next(sample for sample in samples if sample["outputLimit"] == limit) for limit in (128, 512)]
        output = estimate(samples)["decode"]
        self.assertEqual(output["value"], 100)
        self.assertEqual(output["confidence"], "low")
        self.assertEqual(output["samples"], 2)


if __name__ == "__main__":
    unittest.main()
