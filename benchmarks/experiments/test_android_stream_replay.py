"""Reduction of pulled on-device stream traces (issues #226, #357).

Hermetic: a synthetic trace in StreamTrace.toJson()'s shape, no phone, no adb.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import android_stream_replay as asr  # noqa: E402

STOP_TRACE = {"path": "tail", "unfinalized_s": 1.5,
              "totals": {"engine_calls": 1, "engine_audio_s": 1.5, "engine_ms": 900.0}}


def _trace(provenance="LIVE_STREAM", final=True):
    """A 4 s take: sample 0 captured at t=100, chunks every 0.25 s, Stop at 104.0."""
    t0 = 100.0
    sends = [[t0 + (i + 1) * 0.25, (i + 1) * 4000] for i in range(16)]
    events = [
        [101.7, {"type": "partial", "text": "hello", "trace": {"covered_s": 1.0}}],
        [103.2, {"type": "partial", "text": "hello there", "trace": {"covered_s": 2.5}}],
    ]
    trace = {"totals": {"calls": 3, "engine_calls": 3, "engine_audio_s": 5.0,
                        "engine_ms": 2400.0, "preempted": 0, "preempted_ms": 0.0},
             "by_kind": {}, "stop": STOP_TRACE}
    if final:
        events.append([105.1, {"type": "final", "text": "hello there you",
                               "duration_s": 4.0, "trace": trace}])
    else:
        events.append([104.2, {"type": "error", "message": "fallback: engine failed",
                               "trace": trace}])
    return {"version": 1, "t_start": t0, "sends": sends, "events": events,
            "commits": [104.2], "busy_commit_retries": 0, "samples": 64000,
            "prepare_ms": 3.0, "fallback": None if final else "engine failed",
            "marks": {"take": "short", "repeat": 0, "state": "warm", "model": "m.gguf",
                      "stop_pressed": 104.0, "delivered": 105.3,
                      "final_provenance": provenance, "batch_text": "hello there you",
                      "app_cpu_s": 6.0,
                      "thermal_before": {"status": 0}, "thermal_after": {"status": 1}}}


class RunMetricsTest(unittest.TestCase):
    def test_stop_is_measured_from_the_users_stop(self):
        m = asr.run_metrics(_trace(), {"reference": "hello there you"})
        self.assertNotIn("failed", m)
        self.assertEqual(m["stop_to_final_ms"], 1100.0)
        self.assertEqual(m["finish_to_final_ms"], 900.0)
        self.assertEqual(m["stop_to_delivered_ms"], 1300.0)
        self.assertEqual(m["first_partial"]["wall_s"], 1.7)
        self.assertAlmostEqual(m["partial_age_ms"]["p50"], 700.0)
        self.assertEqual(m["work"]["engine_wall_per_audio_s"], 0.6)
        self.assertEqual(m["stop"]["engine_audio_s"], 1.5)
        self.assertEqual(m["app_cpu_per_audio_s"], 1.5)
        self.assertEqual(m["wer_final_vs_ref"], 0.0)
        self.assertTrue(m["audio_complete"])

    def test_a_batch_fallback_is_a_failed_stream(self):
        m = asr.run_metrics(_trace(provenance="BATCH", final=False), {"reference": "hello there you"})
        self.assertIn("failed", m)

    def test_aggregate_reports_sample_counts(self):
        runs = []
        for repeat in range(3):
            m = asr.run_metrics(_trace(), {"reference": "hello there you"})
            runs.append(m)
        agg = asr.aggregate(runs)["short"]
        self.assertEqual(agg["stop_to_final_ms"]["n"], 3)
        self.assertEqual(agg["thermal_status_max"], 1)
        self.assertEqual(agg["app_cpu_per_audio_s_median"], 1.5)


class MergeTest(unittest.TestCase):
    def _part(self, label, **provenance):
        base = {"client": "android-on-device", "device": {"model": "Pixel 10 Pro"}, "model": "m.gguf",
                "cadence": {"min_partial_seconds": 1.0, "partial_interval_seconds": 1.0},
                "warmup": True, "batch": True, "cool_to_c": 33.0, "workload_manifest_sha256": "x",
                "repo_revision": "r", "repeats": [{"repeat": 0}]}
        base.update(provenance)
        run = asr.run_metrics(_trace(), {"reference": "hello there you"})
        return {"label": label, "provenance": base, "runs": [run]}

    def test_parts_of_one_configuration_merge_with_renumbered_repeats(self):
        merged = asr.merge("all", [self._part("a"), self._part("b")])
        self.assertEqual([r["repeat"] for r in merged["runs"]], [0, 1])
        self.assertEqual(merged["aggregate"]["short"]["runs"], 2)

    def test_multi_repeat_parts_are_refused(self):
        with self.assertRaises(asr.RunnerError):
            asr.merge("all", [self._part("a", repeats=[{"repeat": 0}, {"repeat": 1}]), self._part("b")])

    def test_parts_from_another_device_or_batch_setting_are_refused(self):
        for change in ({"device": {"model": "Pixel 8"}}, {"batch": False}):
            with self.assertRaises(asr.RunnerError):
                asr.merge("all", [self._part("a"), self._part("b", **change)])


if __name__ == "__main__":
    unittest.main()
