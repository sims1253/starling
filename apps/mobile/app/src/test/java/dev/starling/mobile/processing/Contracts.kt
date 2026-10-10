package dev.starling.mobile.processing

import java.io.File

/** The shared contract directory (packages/contracts), set by the Gradle test task. */
object Contracts {
    val dir: File = File(
        requireNotNull(System.getProperty("starling.contracts")) {
            "run through Gradle: the starling.contracts system property names packages/contracts"
        },
    )

    fun modeRouting(name: String): String = File(dir, "mode-routing/$name").readText(Charsets.UTF_8)

    fun fixture(name: String): String = modeRouting("fixtures/$name")

    fun insertionBoundaryFixture(name: String): String =
        File(dir, "insertion-boundary/fixtures/$name").readText(Charsets.UTF_8)
}
