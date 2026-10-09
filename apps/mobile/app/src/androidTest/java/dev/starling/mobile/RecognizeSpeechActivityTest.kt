package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.speech.RecognizerIntent
import android.widget.Button
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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

    @Test
    fun popupLaunchesListeningAndShowsTheCallersPrompt() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent("Say a city")).use { scenario ->
            var listening = false
            scenario.onActivity { activity ->
                assertTrue(!activity.isFinishing)
                assertEquals("Say a city", activity.findViewById<TextView>(R.id.recognize_prompt).text.toString())
                listening = activity.findViewById<TextView>(R.id.recognize_status).text.toString() in setOf(
                    activity.getString(R.string.recognize_listening),
                    activity.getString(R.string.recognize_listening_loading),
                )
            }
            scenario.onActivity { it.findViewById<Button>(R.id.recognize_cancel_button).performClick() }
            // An emulator started without audio input (-noaudio) cannot open
            // the microphone: the popup reports RESULT_AUDIO_ERROR then and
            // Cancel keeps it. With a microphone, Cancel is RESULT_CANCELED.
            assertEquals(
                if (listening) Activity.RESULT_CANCELED else RecognizerIntent.RESULT_AUDIO_ERROR,
                scenario.result.resultCode,
            )
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
    fun backCancelsWithoutAResult() {
        ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(recognizeIntent()).use { scenario ->
            scenario.onActivity { activity ->
                @Suppress("DEPRECATION")
                activity.onBackPressed()
            }
            val result = scenario.result
            assertTrue(result.resultCode != Activity.RESULT_OK)
            assertTrue(result.resultData?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS).isNullOrEmpty())
        }
    }
}
