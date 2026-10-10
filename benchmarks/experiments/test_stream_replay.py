"""Paced streaming replay metrics and threshold checks (issues #226, #357).

Hermetic: synthetic event logs only, no server, no websockets package.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_replay as sr  # noqa: E402


def _log(partials, final, *, duration_s=4.0, frame_s=0.5, commit_at=4.1):
    """An event log with frames sent on schedule from t=100."""
    t0 = 100.0
    n = int(duration_s / frame_s)
    sends = [(t0 + (i + 1) * frame_s, int((i + 1) * frame_s * sr.SAMPLE_RATE))
             for i in range(n)]
    events = [(t0 + t, {"type": "partial", **m}) for t, m in partials]
    events.append((t0 + final[0], {"type": "final", **final[1]}))
    return {"t_start": t0, "sends": sends, "events": events,
            "commits": [t0 + commit_at], "busy_commit_retries": 0,
            "samples": int(duration_s * sr.SAMPLE_RATE)}


FINAL_TRACE = {"totals": {"engine_calls": 5, "engine_audio_s": 8.0, "engine_ms": 400.0,
                          "busy": 0, "reused": 1},
               "by_kind": {}, "stop": {"path": "tail", "unfinalized_s": 1.0,
                                       "totals": {"engine_audio_s": 1.0, "engine_ms": 50.0}}}


class WerTest(unittest.TestCase):
    def test_normalized_word_errors(self):
        self.assertEqual(sr.wer("Hello, World", "hello world"), 0.0)
        self.assertAlmostEqual(sr.wer("a b c d", "a x c"), 0.5)
        self.assertEqual(sr.wer("", ""), 0.0)
        self.assertEqual(sr.wer("", "x"), 1.0)


class TakeMetricsTest(unittest.TestCase):
    def test_age_backlog_first_partial_and_stop(self):
        log = _log(
            partials=[
                (1.2, {"text": "", "stable_words": 0, "trace": {"covered_s": 1.0}}),
                (2.3, {"text": "hello there", "stable_words": 1,
                       "trace": {"covered_s": 2.0}}),
                (3.6, {"text": "hello there you", "stable_words": 1,
                       "trace": {"covered_s": 3.0}}),
            ],
            final=(4.4, {"text": "hello there you", "duration_s": 4.0,
                         "trace": FINAL_TRACE}),
        )
        m = sr.take_metrics(log, "hello there you", "hello there you")
        self.assertEqual(m["first_partial"]["wall_s"], 2.3)
        self.assertEqual(m["first_partial"]["audio_sent_s"], 2.0)
        # Ages: 200 ms, 300 ms, 600 ms.
        self.assertAlmostEqual(m["partial_age_ms"]["p50"], 300.0, places=6)
        self.assertAlmostEqual(m["partial_age_ms"]["max"], 600.0, places=6)
        # At t=3.6 the sender had sent 3.5 s; 3.0 s reflected.
        self.assertAlmostEqual(m["backlog_s"]["max"], 0.5, places=6)
        self.assertEqual(m["stop_to_final_ms"], 300.0)
        self.assertTrue(m["audio_complete"])
        self.assertEqual(m["work"]["engine_audio_per_audio_s"], 2.0)
        self.assertEqual(m["stop"]["path"], "tail")
        self.assertEqual(m["stable_violations"], 0)
        self.assertEqual(m["wer_final_vs_ref"], 0.0)

    def test_stable_prefix_violation_and_incomplete_audio(self):
        log = _log(
            partials=[(2.3, {"text": "hello their", "stable_words": 2,
                             "trace": {"covered_s": 2.0}})],
            final=(4.4, {"text": "hello there", "duration_s": 3.5, "trace": FINAL_TRACE}),
        )
        m = sr.take_metrics(log, "hello there", None)
        self.assertEqual(m["stable_violations"], 1)
        self.assertFalse(m["audio_complete"])
        self.assertIsNone(m["wer_final_vs_batch"])

    def test_missing_final_is_a_failure(self):
        log = _log(partials=[], final=(4.4, {"text": "x", "duration_s": 4.0}))
        log["events"] = []
        self.assertEqual(sr.take_metrics(log, "x", None)["failed"], "no final")

    def test_missing_trace_is_missing_work_not_zero(self):
        log = _log(partials=[], final=(4.4, {"text": "x", "duration_s": 4.0}))
        m = sr.take_metrics(log, "x", None)
        self.assertIsNone(m["work"]["engine_audio_per_audio_s"])
        self.assertIsNone(m["work"]["engine_wall_per_audio_s"])
        self.assertIsNone(m["stop"]["engine_audio_s"])
        m.update({"take": "short"})
        ok = sr.take_metrics(_log(partials=[], final=(4.4, {
            "text": "x", "duration_s": 4.0, "trace": FINAL_TRACE})), "x", None)
        ok.update({"take": "short"})
        agg = sr._aggregate([ok, m])["short"]
        self.assertIsNone(agg["engine_wall_per_audio_s_median"])
        self.assertIsNone(agg["engine_audio_per_audio_s_median"])
        self.assertIsNone(agg["stop_engine_audio_s_max"])

    def test_missing_partial_age_is_missing_not_skipped(self):
        with_age = _log(
            partials=[(1.2, {"text": "hi", "stable_words": 0, "trace": {"covered_s": 1.0}})],
            final=(4.4, {"text": "hi", "duration_s": 4.0, "trace": FINAL_TRACE}))
        without = _log(
            partials=[(1.2, {"text": "hi", "stable_words": 0})],
            final=(4.4, {"text": "hi", "duration_s": 4.0, "trace": FINAL_TRACE}))
        runs = []
        for log in (with_age, without):
            m = sr.take_metrics(log, "hi", "hi")
            m.update({"take": "medium"})
            runs.append(m)
        agg = sr._aggregate(runs)["medium"]
        self.assertIsNone(agg["partial_age_ms_p95_median"])
        self.assertIsNone(agg["backlog_s_max"])
        self.assertIsNotNone(sr._aggregate(runs[:1])["medium"]["partial_age_ms_p95_median"])

    def test_one_unmeasured_partial_leaves_the_run_unmeasured(self):
        mixed = _log(
            partials=[(1.2, {"text": "hi", "stable_words": 0, "trace": {"covered_s": 1.0}}),
                      (2.2, {"text": "hi there", "stable_words": 0})],
            final=(4.4, {"text": "hi there", "duration_s": 4.0, "trace": FINAL_TRACE}))
        m = sr.take_metrics(mixed, "hi there", "hi there")
        self.assertIsNone(m["partial_age_ms"]["max"])
        self.assertIsNone(m["backlog_s"]["max"])
        m.update({"take": "medium"})
        agg = sr._aggregate([m])["medium"]
        self.assertIsNone(agg["partial_age_ms_p95_median"])
        self.assertIsNone(agg["backlog_s_max"])

    def test_preempted_previews_count_as_engine_wall_time(self):
        trace = {**FINAL_TRACE, "totals": {**FINAL_TRACE["totals"], "preempted": 2,
                                           "preempted_ms": 400.0}}
        log = _log(partials=[], final=(4.4, {"text": "x", "duration_s": 4.0, "trace": trace}))
        m = sr.take_metrics(log, "x", None)
        self.assertEqual(m["work"]["engine_wall_per_audio_s"], 0.2)  # (400 + 400) ms / 4 s
        self.assertEqual(m["work"]["preempted_calls"], 2)

    def test_stop_latency_counts_busy_commit_retries(self):
        log = _log(partials=[], final=(4.4, {"text": "x", "duration_s": 4.0,
                                             "trace": FINAL_TRACE}))
        log["commits"].append(100.0 + 4.35)  # a busy commit, then a retry
        self.assertEqual(sr.take_metrics(log, "x", None)["stop_to_final_ms"], 300.0)


class AggregateTest(unittest.TestCase):
    def test_a_take_without_partials_has_no_first_partial_latency(self):
        with_partial = _log(
            partials=[(1.2, {"text": "hi", "stable_words": 0, "trace": {"covered_s": 1.0}})],
            final=(4.4, {"text": "hi", "duration_s": 4.0, "trace": FINAL_TRACE}))
        without = _log(partials=[], final=(4.4, {"text": "hi", "duration_s": 4.0,
                                                 "trace": FINAL_TRACE}))
        runs = []
        for log in (with_partial, without):
            m = sr.take_metrics(log, "hi", "hi")
            m.update({"take": "short"})
            runs.append(m)
        agg = sr._aggregate(runs)["short"]
        self.assertEqual(agg["first_partial_missing"], 1)
        self.assertIsNone(agg["first_partial_wall_s_median"])
        self.assertEqual(sr._aggregate(runs[:1])["short"]["first_partial_wall_s_median"], 1.2)


class LocateErrorsTest(unittest.TestCase):
    UTTS = [{"text": "one two three four", "start_s": 0.0, "end_s": 4.0},
            {"text": "five six seven eight", "start_s": 10.0, "end_s": 14.0}]
    REF = "one two three four five six seven eight"
    CALLS = [{"kind": "window", "start_s": 0.0, "end_s": 12.0, "result": "ok"},
             {"kind": "preview", "start_s": 9.0, "end_s": 11.0, "result": "ok"},
             {"kind": "flush_tail", "start_s": 9.0, "end_s": 14.0, "result": "ok"},
             {"kind": "redecode", "start_s": 7.5, "end_s": 14.0, "result": "ok"}]

    def test_omission_placed_in_the_take(self):
        out = sr.locate_errors(self.REF, "one two three four seven eight", self.UTTS,
                               self.REF, self.CALLS)
        self.assertEqual((out["omitted_words"], out["inserted_words"]), (2, 0))
        (span,) = out["spans"]
        self.assertEqual(span["omitted"], "five six")
        # "five" is spread over 10-14 s: 10.5 s, inside the overlap of the
        # window with the tail commitment, which spans its re-decode from
        # 7.5 s (previews do not count).
        self.assertEqual((span["t_s"], span["where"], span["overlap_s"]),
                         (10.5, "overlap", [7.5, 12.0]))
        # Candidates of one commitment do not overlap each other: a first
        # window re-decoded shorter leaves 2.5 s outside any overlap.
        first = [{"kind": "window", "start_s": 0.0, "end_s": 12.0},
                 {"kind": "redecode", "start_s": 0.0, "end_s": 10.5}]
        out = sr.locate_errors("one two three four", "one three four",
                               [{"text": "one two three four", "start_s": 0.0, "end_s": 10.0}],
                               "one two three four", first)
        self.assertEqual((out["spans"][0]["where"], out["spans"][0]["overlap_s"]),
                         ("window", None))

    def test_duplicated_run_and_far_errors(self):
        out = sr.locate_errors(self.REF, "One, two two three four five six seven eight",
                               self.UTTS, self.REF, self.CALLS)
        self.assertEqual((out["inserted_words"], out["duplicated_words"]), (1, 1))
        self.assertEqual(out["spans"][0]["where"], "window")  # 1.5 s: far from 9-12 s
        out = sr.locate_errors(self.REF, "one two three four five six seven eight nine",
                               self.UTTS, self.REF, self.CALLS)
        self.assertEqual((out["inserted_words"], out["duplicated_words"]), (1, 0))
        self.assertEqual(sr.locate_errors(self.REF, self.REF, self.UTTS, self.REF,
                                          self.CALLS)["spans"], [])

    def test_take_metrics_reports_vs_batch_and_loops(self):
        loop = "a little bit of " * 5
        log = _log(partials=[(1.2, {"text": "hello " + loop, "trace": {"covered_s": 1.0}}),
                             (2.3, {"text": "hello there", "trace": {"covered_s": 2.0}})],
                   final=(4.4, {"text": "hello there", "duration_s": 4.0, "trace": FINAL_TRACE}))
        m = sr.take_metrics(log, "hello there you", "hello there you",
                            [{"text": "hello there you", "start_s": 0.0, "end_s": 3.0}])
        self.assertEqual(m["vs_batch"]["omitted_words"], 1)
        self.assertEqual((m["looping_partials"], m["longest_partial_loop_words"]), (1, 20))
        agg = sr._aggregate([{**m, "take": "short"}])["short"]
        self.assertEqual((agg["omitted_vs_batch_median"], agg["looping_partials_total"]), (1, 1))
        self.assertIsNone(sr._aggregate([{**sr.take_metrics(log, "hello", None),
                                          "take": "x"}])["x"]["omitted_vs_batch_median"])

    def test_longest_loop_needs_four_repeats(self):
        self.assertEqual(sr.longest_loop("a b a b a b a b c".split()), 8)
        self.assertEqual(sr.longest_loop("no no no no no".split()), 0)  # under 8 words
        self.assertEqual(sr.longest_loop("x y z x y z x y z".split()), 0)  # three repeats


class CheckTest(unittest.TestCase):
    def test_rules(self):
        prov = {"provenance": {"workload_manifest_sha256": "w1"}}
        base = {**prov, "aggregate": {"short": {"a": 10.0, "b": 0.1, "c": True}}}
        cand = {**prov, "aggregate": {"short": {"a": 12.0, "b": 0.12, "c": False}}}
        rules = {"rules": [
            {"take": "short", "metric": "a", "max_vs_baseline_ratio": 1.1},
            {"take": "short", "metric": "b", "max_vs_baseline_delta": 0.05},
            {"take": "short", "metric": "c", "equals": True},
            {"take": "short", "metric": "a", "max": 15},
            {"take": "long", "metric": "a", "max": 1},
        ]}
        verdicts = [r["pass"] for r in sr.check(rules, base, cand)]
        self.assertEqual(verdicts, [False, True, False, True, False])

    def test_different_workloads_are_not_comparable(self):
        base = {"provenance": {"workload_manifest_sha256": "w1"},
                "aggregate": {"short": {"a": 1.0}}}
        rules = {"rules": [{"take": "short", "metric": "a", "max": 2}]}
        cand = {**base, "provenance": {"workload_manifest_sha256": "w2"}}
        out = sr.check(rules, base, cand)
        self.assertEqual([r["pass"] for r in out], [False, True])
        self.assertEqual(out[0]["metric"], "workload_manifest_sha256")
        self.assertFalse(sr.check(rules, {"aggregate": base["aggregate"]}, cand)[0]["pass"])
        self.assertEqual([r["pass"] for r in sr.check(rules, base, base)], [True])


class ReplayReceiverTest(unittest.TestCase):
    def test_receiver_failure_fails_the_run_with_its_cause(self):
        import types

        class Boom(Exception):
            pass

        class FakeWs:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                raise Boom("connection dropped")

            def send(self, _msg):
                pass

        client = types.ModuleType("websockets.sync.client")
        client.connect = lambda *a, **k: FakeWs()
        saved = {k: sys.modules.get(k) for k in
                 ("websockets", "websockets.sync", "websockets.sync.client")}
        sys.modules.update({"websockets": types.ModuleType("websockets"),
                            "websockets.sync": types.ModuleType("websockets.sync"),
                            "websockets.sync.client": client})
        try:
            with self.assertRaises(sr.RunnerError) as cm:
                sr.replay("ws://x", b"\0\0" * 3200, 100.0, timeout_s=30.0)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertIn("connection dropped", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
