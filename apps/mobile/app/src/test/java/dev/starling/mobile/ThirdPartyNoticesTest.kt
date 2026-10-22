package dev.starling.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ThirdPartyNoticesTest {
    @Test
    fun componentsResolveTheirTexts() {
        val notices = ThirdPartyNotices.parse(
            """
            {"schema":1,"app":"android","starling":{"license":"MIT","text":"own"},
             "components":[{"name":"androidx.core:core","version":"1.19.1","license":"Apache-2.0",
                            "url":"https://developer.android.com","texts":["apache","missing"]}],
             "texts":{"own":"MIT License","apache":"Apache License"}}
            """.trimIndent(),
        )
        assertEquals("MIT License", notices.starlingLicense)
        val core = notices.components.single()
        assertEquals("androidx.core:core", core.name)
        assertEquals(listOf("Apache License"), core.texts)
    }

    @Test
    fun noticesWithoutStarlingsLicenseAreRefused() {
        assertThrows(IllegalArgumentException::class.java) {
            ThirdPartyNotices.parse("""{"starling":{"text":"gone"},"components":[],"texts":{}}""")
        }
    }
}
