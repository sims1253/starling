package dev.starling.mobile

import android.content.ComponentName
import android.content.Intent
import android.speech.RecognitionService
import android.speech.SpeechRecognizer
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.starling.mobile.engine.OnDeviceEngine
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
