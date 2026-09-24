package dev.starling.mobile

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.engine.ModelCatalog
import dev.starling.mobile.engine.ModelDownloader
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.InferenceResult
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * End to end on the real engine: download the recommended model with the
 * app's own downloader, adopt it, and transcribe a LibriSpeech clip.
 *
 * Opt-in because it fetches 553 MB and needs a capable arm64 phone:
 *   ./gradlew connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.e2e=true
 * Gradle uninstalls the test app afterwards, so every run downloads again.
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
            val adopted = engine.adoptDownloaded(done.file)
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
        assertEquals(normalize(REFERENCE), normalize(text))
    }

    private fun normalize(text: String) =
        text.lowercase().replace(Regex("[^a-z' ]"), " ").split(' ').filter { it.isNotEmpty() }.joinToString(" ")

    private companion object {
        // LibriSpeech 2086-149220-0033 (CC BY 4.0), as in benchmarks/wer.py.
        const val REFERENCE =
            "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. " +
                "It is certainly very like the old portrait."
    }
}
