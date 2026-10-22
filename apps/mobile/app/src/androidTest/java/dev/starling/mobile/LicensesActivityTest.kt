package dev.starling.mobile

import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The Licenses screen on a device: the build's generated notices ship in the
 * APK with a text for every component, and a row opens its license.
 */
@RunWith(AndroidJUnit4::class)
class LicensesActivityTest {
    private val context = ApplicationProvider.getApplicationContext<android.app.Application>()

    @Test
    fun theApkCarriesNoticesForWhatItShips() {
        val notices = ThirdPartyNotices.parse(
            context.assets.open(ThirdPartyNotices.ASSET).bufferedReader().use { it.readText() },
        )
        assertTrue(notices.starlingLicense.startsWith("MIT License"))
        val names = notices.components.map { it.name }
        for (expected in listOf("androidx.core:core", "com.squareup.okhttp3:okhttp", "ggml")) {
            assertTrue("$expected listed in $names", expected in names)
        }
        for (component in notices.components) {
            assertTrue("${component.name} has a license text", component.texts.isNotEmpty())
        }
    }

    @Test
    fun aRowOpensItsLicense() {
        ActivityScenario.launch(LicensesActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val row = rows(activity.window.decorView).first {
                    it.contentDescription?.startsWith("ggml,") == true
                }
                val body = row.getChildAt(2) as TextView
                assertEquals(View.GONE, body.visibility)
                row.performClick()
                assertEquals(View.VISIBLE, body.visibility)
                assertTrue(body.text.contains("The ggml authors"))
                row.performClick()
                assertEquals(View.GONE, body.visibility)
            }
        }
    }

    private fun rows(view: View): List<LinearLayout> = when {
        view is LinearLayout && view.isClickable -> listOf(view)
        view is ViewGroup -> (0 until view.childCount).flatMap { rows(view.getChildAt(it)) }
        else -> emptyList()
    }
}
