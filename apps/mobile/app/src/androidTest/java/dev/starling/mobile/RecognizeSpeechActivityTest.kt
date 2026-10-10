package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.speech.RecognizerIntent
import android.view.KeyEvent
import android.widget.Button
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The `ACTION_RECOGNIZE_SPEECH` entry point (E22) as another app sees it:
 * the system resolves the action to Starling's popup, the popup launches and
 * starts listening, and Cancel (button or back) hands RESULT_CANCELED back.
 * The microphone is granted up front so no permission dialog covers the popup.
 */
@RunWith(AndroidJUnit4::class)
class RecognizeSpeechActivityTest {
    @get:Rule
    val microphone: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)

    private val context = ApplicationProvider.getApplicationContext<android.app.Application>()

    private fun recognizeIntent(prompt: String? = null) =
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .setPackage(context.packageName)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .apply { if (prompt != null) putExtra(RecognizerIntent.EXTRA_PROMPT, prompt) }

    @Test
    fun recognizeSpeechResolvesToThePopup() {
        val activities = context.packageManager.queryIntentActivities(
            recognizeIntent(),
            PackageManager.MATCH_DEFAULT_ONLY,
        )
        assertEquals(
            listOf(ComponentName(context, RecognizeSpeechActivity::class.java).className),
            activities.map { it.activityInfo.name },
        )
        assertTrue(activities.single().activityInfo.exported)
    }

    /**
     * Whether the popup opened the microphone. An emulator started with
     * -noaudio may refuse it; the status text is no signal, since a live
     * stream that fails to connect rewrites it while the capture runs.
     */
    private fun listening(scenario: ActivityScenario<RecognizeSpeechActivity>): Boolean {
        var listening = false
        scenario.onActivity { listening = it.capturing }
        return listening
    }

    @Test
    fun popupLaunchesAndShowsTheCallersPrompt() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent("Say a city")).use { scenario ->
            scenario.onActivity { activity ->
                assertTrue(!activity.isFinishing)
                assertEquals("Say a city", activity.findViewById<TextView>(R.id.recognize_prompt).text.toString())
            }
            scenario.onActivity { it.findViewById<Button>(R.id.recognize_cancel_button).performClick() }
            assertTrue(scenario.result.resultCode != Activity.RESULT_OK)
        }
    }

    @Test
    fun cancelWhileListeningReturnsCanceled() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent()).use { scenario ->
            assumeTrue("no microphone on this emulator", listening(scenario))
            scenario.onActivity { it.findViewById<Button>(R.id.recognize_cancel_button).performClick() }
            assertEquals(Activity.RESULT_CANCELED, scenario.result.resultCode)
        }
    }

    @Test
    fun microphoneFailureIsAnAudioError() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent()).use { scenario ->
            assumeTrue("this emulator has a microphone", !listening(scenario))
            // The error stays on screen; closing it keeps the error code.
            scenario.onActivity { it.findViewById<Button>(R.id.recognize_done_button).performClick() }
            assertEquals(RecognizerIntent.RESULT_AUDIO_ERROR, scenario.result.resultCode)
        }
    }

    @Test
    fun overlongPromptIsBounded() {
        val prompt = "x".repeat(1000)
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent(prompt)).use { scenario ->
            scenario.onActivity { activity ->
                val shown = activity.findViewById<TextView>(R.id.recognize_prompt).text.toString()
                assertEquals(200, shown.length)
            }
            scenario.onActivity { it.findViewById<Button>(R.id.recognize_cancel_button).performClick() }
        }
    }

    @Test
    fun systemBackCancelsWithoutAResult() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent()).use { scenario ->
            assumeTrue("no microphone on this emulator", listening(scenario))
            // The key goes to the focused window; the dialog gets focus a
            // moment after it is resumed.
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            val deadline = System.currentTimeMillis() + 10_000
            var focused = false
            while (!focused && System.currentTimeMillis() < deadline) {
                scenario.onActivity { focused = it.hasWindowFocus() }
                if (!focused) Thread.sleep(100)
            }
            assertTrue("the popup never got window focus", focused)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            val result = scenario.result
            assertEquals(Activity.RESULT_CANCELED, result.resultCode)
            assertTrue(result.resultData?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS).isNullOrEmpty())
        }
    }
}
