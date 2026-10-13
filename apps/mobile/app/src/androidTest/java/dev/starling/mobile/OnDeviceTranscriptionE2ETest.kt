package dev.starling.mobile

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.engine.ModelCatalog
import dev.starling.mobile.engine.ModelDownloader
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.transcription.InferenceResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * End to end on the real engine: download the recommended model with the
 * app's own downloader, adopt it, and transcribe a LibriSpeech clip.
 *
 * Opt-in because it fetches 364 MB and needs a capable arm64 phone:
 *   ./gradlew connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.e2e=true
 * Gradle uninstalls the test app afterwards, so every run downloads again.
 *
 * [installedModelTranscribesLibriSpeechClip] runs any other model family the
 * same way, from a GGUF placed in the app's files/e2e-models beforehand
 * (connectedAndroidTest would uninstall it, so install the APKs and run
 * `am instrument` directly):
 *   adb shell run-as dev.starling.mobile.debug mkdir -p files/e2e-models
 *   adb shell run-as dev.starling.mobile.debug cp /data/local/tmp/m.gguf files/e2e-models/
 *   adb shell am instrument -w -e model m.gguf \
 *     -e class dev.starling.mobile.OnDeviceTranscriptionE2ETest#installedModelTranscribesLibriSpeechClip \
 *     dev.starling.mobile.debug.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceTranscriptionE2ETest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Test
    fun recommendedModelTranscribesLibriSpeechClip() {
        assumeTrue(
            "set the e2e instrumentation argument to run",
            InstrumentationRegistry.getArguments().getString("e2e") == "true",
        )
        val context = instrumentation.targetContext
        val engine = OnDeviceEngine(File(context.filesDir, "e2e-model"))
        val spec = ModelCatalog.RECOMMENDED_PARAKEET

        if (!engine.hasModel()) {
            val result = ModelDownloader().download(spec, engine.downloadFile(spec)) { _, _, _ -> }
            val done = result as? ModelDownloader.Result.Done
                ?: throw AssertionError("download failed: $result")
            val adopted = engine.adoptDownloaded(done.file, spec.fileName)
            if (adopted !is OnDeviceEngine.ImportResult.Imported) throw AssertionError("model rejected: $adopted")
        }

        val wav = File(context.cacheDir, "e2e.wav")
        instrumentation.context.assets.open("librispeech-2086-149220-0033.wav").use { input ->
            wav.outputStream().use { input.copyTo(it) }
        }
        val result = engine.transcribe(wav)
        val text = (result as? InferenceResult.Success)?.rawTranscript
            ?: throw AssertionError("transcription failed: $result")
        Log.i("StarlingE2E", "transcript: $text")
        assertEquals("transcript mismatch, got: $text", normalize(REFERENCE), normalize(text))
    }

    @Test
    fun installedModelTranscribesLibriSpeechClip() {
        val model = InstrumentationRegistry.getArguments().getString("model")
        assumeTrue("set the model instrumentation argument to run", model != null)
        val context = instrumentation.targetContext
        // Same TPU artifact directory as the app (scripts/npu/npu_compile.sh --app).
        val engine = OnDeviceEngine(File(context.filesDir, "e2e-models"), npuDirs = listOf(File(context.filesDir, "npu")))
        assumeTrue("$model is not in files/e2e-models", engine.installedModels().any { it.name == model })
        val wav = File(context.cacheDir, "e2e.wav")
        instrumentation.context.assets.open("librispeech-2086-149220-0033.wav").use { input ->
            wav.outputStream().use { input.copyTo(it) }
        }
        // The first call loads and warms the model; the second is the steady state.
        for (run in 0 until 2) {
            val started = System.nanoTime()
            val result = engine.transcribe(wav, model)
            val millis = (System.nanoTime() - started) / 1_000_000
            val text = (result as? InferenceResult.Success)?.rawTranscript
                ?: throw AssertionError("transcription failed: $result")
            Log.i("StarlingE2E", "$model run $run: ${millis} ms: $text")
            val expected = normalize(REFERENCE).split(' ')
            val got = normalize(text).split(' ').toSet()
            val matched = expected.count { it in got }
            // Models differ in spelling ("any more"/"anymore") and casing; a
            // wrong engine or garbled audio path matches almost nothing.
            assertTrue("transcript too far from the reference: $text", matched >= expected.size * 8 / 10)
        }
        engine.releaseWhenIdle()
    }

    private fun normalize(text: String) =
        text.lowercase().replace(NON_LETTERS, " ").split(' ').filter { it.isNotEmpty() }.joinToString(" ")

    private companion object {
        val NON_LETTERS = Regex("[^a-z' ]")

        // LibriSpeech 2086-149220-0033 (CC BY 4.0), as in benchmarks/fast_engine/wer.py.
        const val REFERENCE =
            "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. " +
                "It is certainly very like the old portrait."
    }
}
