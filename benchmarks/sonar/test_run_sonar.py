"""CPU-only harness regressions; no model weights, downloads, or GPU required."""

from contextlib import ExitStack, contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_sonar as runner
from variants import StarlingVariant


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tsv = self.root / "test.tsv"
        self.tsv.write_text("audio_path\ttranscription\nclip.wav\thello\n", encoding="utf-8")
        self.args = ["--tsv", str(self.tsv), "--results-root", str(self.root), "--tag", "run"]
        self.variant = StarlingVariant(
            "test_cpu", "parakeet", self.root / "model.gguf", "cpu", Path("serve")
        )
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def test_prepare_only_does_not_require_inference_files(self):
        with patch.object(runner, "build_variants") as variants:
            self.assertEqual(runner.main([*self.args, "--prepare-only"]), 0)
        variants.assert_not_called()

    def test_force_prepare_refreshes_cached_split(self):
        data = self.root / "data"
        target = data / "en" / "fleurs" / "test.tsv"
        target.parent.mkdir(parents=True)
        target.write_text("cached", encoding="utf-8")

        def discover(args):
            self.assertEqual(args[-2:], ["--max-samples", "0"])
            target.write_text("all samples", encoding="utf-8")
            return 0

        with patch.object(runner, "_sonar_cli", side_effect=discover) as cli:
            runner._ensure_dataset("en", "fleurs", 0, data, force=False)
            cli.assert_not_called()
            runner._ensure_dataset("en", "fleurs", 0, data, force=True)
            cli.assert_called_once()
        self.assertEqual(target.read_text(encoding="utf-8"), "all samples")

    def test_existing_results_are_preserved_and_rejected(self):
        previous = self.root / "run"
        previous.mkdir()
        scores = previous / "scores_old.json"
        scores.write_text("previous results", encoding="utf-8")
        with (
            patch.object(runner, "build_variants", return_value=[self.variant]),
            patch.object(runner, "_start_server") as start,
        ):
            with self.assertRaisesRegex(SystemExit, "Results already exist"):
                runner.main(self.args)
            start.assert_not_called()
        self.assertEqual(scores.read_text(encoding="utf-8"), "previous results")

    def test_sweep_reports_partial_failure_and_keeps_successful_rows(self):
        rows = [{"model_name": self.variant.sonar_name, "wer": 0.0}]
        variants = [self.variant, self.variant]
        with (
            patch.object(runner, "build_variants", return_value=variants),
            patch.object(runner, "_start_server", return_value=(Mock(), "http://localhost")),
            patch.object(runner, "_stop_server") as stop,
            patch.object(
                runner, "_evaluate_variant", side_effect=[{}, RuntimeError("failed variant")]
            ),
            patch.object(runner, "_render_leaderboard", return_value=rows),
        ):
            self.assertEqual(runner.main([*self.args, "--no-warmup"]), 1)
            self.assertEqual(stop.call_count, 2)
        manifest = json.loads((self.root / "run" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest[0]["model_name"], self.variant.sonar_name)

    def test_successful_sweep_requires_leaderboard_rows(self):
        for rows, expected in [([], 1), ([{"model_name": self.variant.sonar_name}], 0)]:
            with (
                self.subTest(rows=rows),
                patch.object(runner, "build_variants", return_value=[self.variant]),
                patch.object(runner, "_start_server", return_value=(Mock(), "http://localhost")),
                patch.object(runner, "_stop_server"),
                patch.object(runner, "_evaluate_variant", return_value={}),
                patch.object(runner, "_render_leaderboard", return_value=rows),
            ):
                args = [*self.args, "--tag", f"rows-{expected}", "--no-warmup"]
                self.assertEqual(runner.main(args), expected)

    def test_gpu_lock_covers_spawn_evaluation_and_shutdown(self):
        variant = StarlingVariant(
            "test_vulkan", "parakeet", self.root / "model.gguf", "vulkan", Path("serve")
        )
        events = []

        @contextmanager
        def lock(**kwargs):
            self.assertEqual(kwargs["uuid"], "vulkan:test-device")
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        def start(*args, **kwargs):
            self.assertEqual(kwargs["gpu_uuid"], "vulkan:test-device")
            events.append("spawn")
            return Mock(), "http://localhost"

        def evaluate(*args, **kwargs):
            events.append("evaluate")
            raise RuntimeError("evaluation failed")

        with (
            patch.object(runner, "build_variants", return_value=[variant]),
            patch.object(runner, "with_gpu_lock", side_effect=lock),
            patch.object(runner, "_start_server", side_effect=start),
            patch.object(runner, "_evaluate_variant", side_effect=evaluate),
            patch.object(runner, "_stop_server", side_effect=lambda proc: events.append("stop")),
            patch.object(runner, "_render_leaderboard", return_value=[]),
        ):
            self.assertEqual(
                runner.main([*self.args, "--no-warmup", "--gpu-uuid", "vulkan:test-device"]), 1
            )
        self.assertEqual(events, ["lock", "spawn", "evaluate", "stop", "unlock"])

    def test_runtime_backend_mismatch_stops_server(self):
        variant = StarlingVariant(
            "test_vulkan", "parakeet", self.root / "model.gguf", "vulkan", Path("serve")
        )
        log = self.root / "server.log"
        proc = Mock()

        def ready(*args):
            log.write_text(
                "[starling-serve] starting on 127.0.0.1:1234 (model=parakeet, backend=CPU, abi=6)\n",
                encoding="utf-8",
            )

        with (
            patch.object(runner, "spawn_gpu_subprocess", return_value=proc) as spawn,
            patch.object(runner, "_wait_healthy", side_effect=ready),
            patch.object(runner, "_stop_server") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "requested backend 'vulkan'.*'cpu'"):
                runner._start_server(
                    variant, log_path=log, timeout_s=1, gpu_uuid="vulkan:test-device"
                )
            self.assertEqual(spawn.call_args.kwargs["uuid"], "vulkan:test-device")
            stop.assert_called_once_with(proc)

    def test_cpu_variant_overrides_inherited_device(self):
        log = self.root / "server.log"

        def ready(*args):
            log.write_text(
                "[starling-serve] starting on 127.0.0.1:1234 (model=parakeet, backend=CPU, abi=6)\n",
                encoding="utf-8",
            )

        with (
            patch.dict(os.environ, {"STARLING_GGML_DEVICE": "Vulkan0"}),
            patch.object(runner.subprocess, "Popen") as spawn,
            patch.object(runner, "_wait_healthy", side_effect=ready),
        ):
            runner._start_server(self.variant, log_path=log, timeout_s=1)
            self.assertEqual(spawn.call_args.kwargs["env"]["STARLING_GGML_DEVICE"], "cpu")

    def test_vulkan_runtime_device_is_accepted(self):
        variant = StarlingVariant(
            "test_vulkan", "parakeet", self.root / "model.gguf", "vulkan", Path("serve")
        )
        log = self.root / "server.log"
        proc = Mock()

        def ready(*args):
            log.write_text(
                "[starling-serve] starting on 127.0.0.1:1234 (model=parakeet, backend=Vulkan0, abi=6)\n",
                encoding="utf-8",
            )

        with (
            patch.object(runner, "spawn_gpu_subprocess", return_value=proc),
            patch.object(runner, "_wait_healthy", side_effect=ready),
        ):
            actual, _ = runner._start_server(
                variant, log_path=log, timeout_s=1, gpu_uuid="vulkan:test-device"
            )
        self.assertIs(actual, proc)

    def test_dependency_stack_uses_cpu_torch(self):
        import torch
        import torchaudio
        import torchvision

        self.assertIsNone(torch.version.cuda)
        self.assertTrue(torchaudio.__version__)
        self.assertTrue(torchvision.__version__)

    def test_quality_bypass_disables_real_sonar_prewarm(self):
        from psdn_sonar import quality_models
        from psdn_sonar.evaluators import single_speaker

        evaluator = single_speaker.SingleSpeakerEvaluator
        with ExitStack() as stack:
            # Restore SONAR internals after the opt-in patch is exercised.
            stack.enter_context(patch.object(evaluator, "_compute_audio_quality"))
            names = ("_get_dnsmos", "_get_utmos", "_get_squim")
            for name in names:
                stack.enter_context(
                    patch.object(
                        quality_models,
                        name,
                        side_effect=AssertionError(f"{name} was not disabled"),
                    )
                )
            runner._disable_audio_quality()
            loaders = [
                stack.enter_context(
                    patch.object(quality_models, name, wraps=getattr(quality_models, name))
                )
                for name in names
            ]
            path, quality = evaluator._compute_audio_quality({"audio_path": "clip.wav"})
            self.assertEqual(path, "clip.wav")
            self.assertEqual(quality, single_speaker._EMPTY_AUDIO_QUALITY)

            # Execute the actual upstream prewarm target synchronously. This
            # detects upstream changes to the private integration contract.
            def thread(*, target, daemon):
                return Mock(start=target)

            stack.enter_context(patch("threading.Thread", side_effect=thread))
            with self.assertRaisesRegex(ValueError, "None of the requested models"):
                evaluator.run_evaluation(
                    tsv_path=str(self.tsv),
                    output_dir=str(self.root / "quality"),
                    models=[],
                    compute_sem=False,
                    language="en",
                )
            for loader in loaders:
                loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
