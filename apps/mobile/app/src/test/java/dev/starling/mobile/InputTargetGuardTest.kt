package dev.starling.mobile

import dev.starling.mobile.ui.InputTargetGuard
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InputTargetGuardTest {
    @Test
    fun lateTranscriptCannotCrossEditorTarget() {
        val guard = InputTargetGuard<Any>()
        val firstTarget = Any()
        val secondTarget = Any()

        guard.targetStarted(firstTarget)
        val snapshot = guard.capture()!!
        assertTrue(guard.isCurrent(snapshot, firstTarget))

        guard.targetStarted(secondTarget)
        assertFalse(guard.isCurrent(snapshot, secondTarget))
        assertFalse(guard.isCurrent(snapshot, firstTarget))
    }

    @Test
    fun finishingEditorInvalidatesPendingTranscript() {
        val guard = InputTargetGuard<Any>()
        val target = Any()
        guard.targetStarted(target)
        val snapshot = guard.capture()!!
        guard.targetFinished()
        assertFalse(guard.isCurrent(snapshot, target))
    }
}
