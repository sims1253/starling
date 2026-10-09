package dev.starling.mobile

import android.content.ComponentName
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.view.inputmethod.InputMethod
import android.speech.RecognitionService
import android.speech.SpeechRecognizer
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.engine.StarlingNative
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayInputStream
import java.io.File

/**
 * Emulator checks for the platform wiring unit tests cannot see: the system
 * must discover the recognition service, and a bad model import must report
 * a reason instead of crashing.
 */
@RunWith(AndroidJUnit4::class)
class DeviceIntegrationTest {
    private val context = ApplicationProvider.getApplicationContext<android.app.Application>()

    @Test
    fun recognitionServiceIsDiscoverableBySystemPickers() {
        val services = context.packageManager.queryIntentServices(
            Intent(RecognitionService.SERVICE_INTERFACE).setPackage(context.packageName),
            0,
        )
        assertEquals(
            listOf(ComponentName(context, StarlingRecognitionService::class.java).className),
            services.map { it.serviceInfo.name },
        )
        assertTrue(SpeechRecognizer.isRecognitionAvailable(context))
    }

    @Test
    fun voiceKeyboardIsDiscoverableAsAnInputMethod() {
        val services = context.packageManager.queryIntentServices(
            Intent(InputMethod.SERVICE_INTERFACE).setPackage(context.packageName),
            0,
        )
        assertEquals(
            listOf(ComponentName(context, VoiceInputService::class.java).className),
            services.map { it.serviceInfo.name },
        )
    }

    @Test
    fun takeServiceIsAMicrophoneForegroundService() {
        // Android 14+ refuses a background microphone without this type.
        val info = context.packageManager.getServiceInfo(
            ComponentName(context, CaptureForegroundService::class.java),
            0,
        )
        assertFalse(info.exported)
        if (Build.VERSION.SDK_INT >= 29) {
            assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE, info.foregroundServiceType)
        }
    }

    @Test
    fun packagedEngineMatchesTheExpectedAbi() {
        // Loads the real libstarling_jni: a mismatch here fails every
        // on-device recording at runtime.
        assertEquals(StarlingNative.EXPECTED_ABI_VERSION, StarlingNative.abiVersion())
    }

    @Test
    fun invalidModelImportIsRejectedWithAReason() {
        val dir = File(context.cacheDir, "import-test").apply { deleteRecursively(); mkdirs() }
        try {
            val result = OnDeviceEngine(dir).importModel(ByteArrayInputStream(ByteArray(64)))
            val rejected = result as? OnDeviceEngine.ImportResult.Rejected
            assertTrue("expected rejection, got $result", rejected != null)
            assertTrue(rejected?.reason?.isNotBlank() == true)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun mainActivityLaunches() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity -> assertFalse(activity.isFinishing) }
        }
    }
}
