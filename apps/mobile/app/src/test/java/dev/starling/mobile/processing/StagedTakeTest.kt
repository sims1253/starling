package dev.starling.mobile.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class StagedTakeTest {
    private val commands = SpokenCommands(Contracts.modeRouting("spoken-commands.json"))

    // The keyboard's shipped modes; the Gradle test runs in the module directory.
    private val catalog = ModeCatalog(
        ProfilesDocument.parse(File("src/main/assets/modes/android-profiles.json").readText()),
        commands,
        SpokenInstructions(Contracts.modeRouting("spoken-instructions.json"), commands),
    )

    private fun take(mode: String = "draft") = StagedTake(catalog, catalog.mode(mode), "take-1")

    private fun StagedTake.dictate(vararg partials: String, final: String, attempt: String = "a-${partials.size}") {
        beginSegment()
        partials.forEach(::partial)
        final(attempt, final)
    }

    @Test
    fun theShippedModesFollowTheContract() {
        assertEquals("direct", catalog.mode(null).id)
        assertEquals("direct", catalog.mode("no-longer-shipped").id)
        assertEquals("staged", catalog.mode("draft").processingDelivery)
        assertEquals("insert_enter", catalog.mode("message").delivery)
    }

    @Test
    fun partialsAreTheLiveTailAndProcessingWaitsForTheFinal() {
        val take = take()
        take.beginSegment()
        take.partial("hello comma")
        assertTrue(take.recording)
        assertNull(take.proposal)
        assertEquals("hello comma", take.displayText())
        take.final("a-1", "hello comma world")
        assertFalse(take.recording)
        assertEquals("hello, world", take.proposal?.text)
        assertEquals("hello, world", take.displayText())
        // Raw recovery is one toggle away and never altered.
        take.toggleView()
        assertEquals("hello comma world", take.displayText())
        assertEquals("hello comma world", take.draft.rawText())
    }

    @Test
    fun deliveredTextIsWhatIsOnScreen() {
        val processed = take().apply { dictate(final = "milk comma eggs") }
        assertEquals("milk, eggs", processed.deliveryText())
        val raw = take().apply { dictate(final = "milk comma eggs"); toggleView() }
        assertEquals("milk comma eggs", raw.deliveryText())
    }

    @Test
    fun aLeadingPhraseSwitchesTheModeAndNeverReachesTheField() {
        val take = take("direct")
        take.dictate(final = "Message mode: see you at six period")
        assertEquals("message", take.mode.id)
        assertEquals(StagedTake.Decision.Phrase("message mode", catalog.mode("message")), take.decision)
        assertEquals("see you at six.", take.deliveryText())
        // The raw recognition keeps the phrase.
        assertEquals("Message mode: see you at six period", take.draft.rawText())
    }

    @Test
    fun thatWasLiteralKeepsThePhraseAsWords() {
        val take = take("draft")
        take.dictate(final = "message mode is broken comma again")
        assertEquals("message", take.mode.id)
        take.undoDecision()
        assertEquals("draft", take.mode.id)
        assertNull(take.decision)
        assertEquals("message mode is broken, again", take.deliveryText())
    }

    @Test
    fun theLiteralEscapeKeepsTheRestVerbatim() {
        val take = take("draft")
        take.dictate(final = "literal draft mode comma")
        assertEquals("verbatim", take.mode.id)
        assertEquals(StagedTake.Decision.Literal, take.decision)
        assertNull(take.proposal)
        assertEquals("draft mode comma", take.deliveryText())
    }

    @Test
    fun aTrailingInstructionIsHeldOutOfTheText() {
        val take = take()
        take.dictate(final = "Please send the report Starling, make it formal")
        assertEquals(StagedTake.Decision.Instruction("make it formal"), take.decision)
        assertEquals("Please send the report", take.deliveryText())
        take.undoDecision()
        assertEquals("Please send the report Starling, make it formal", take.deliveryText())
    }

    @Test
    fun aSecondCaptureAppendsWithASpace() {
        val take = take()
        take.dictate(final = "first part")
        take.dictate("second", final = "second part period", attempt = "a-2")
        assertEquals("first part second part.", take.displayText())
        assertEquals("first part second part period", take.draft.rawText())
    }

    @Test
    fun deleteWordEditsTheTextOnScreen() {
        val take = take()
        take.dictate(final = "one comma two three")
        // On screen: "one, two three"; select "two" (code points 5..7).
        take.selectWordAt(6)
        assertEquals(5..7, take.selection)
        take.deleteWord()
        assertEquals("one, three", take.displayText())
        assertEquals("one, three", take.deliveryText())
        take.deleteWord()
        assertEquals("one,", take.displayText().trimEnd())
        // Back to raw restores the recognition and processes it again.
        take.backToRaw()
        assertEquals("one comma two three", take.displayText())
    }

    @Test
    fun wordsAreCodePoints() {
        val text = "😀 hi there"
        assertEquals(2..3, StagedTake.wordAt(text, 2))
        assertEquals(5..9, StagedTake.lastWord(text))
        assertNull(StagedTake.wordAt(text, 1))
    }

    @Test
    fun deliveryIsExactlyOncePerRevisionAndTarget() {
        val take = take()
        take.dictate(final = "hello")
        take.deliveryText()
        assertTrue(take.deliver("d-1", "field-7"))
        assertFalse(take.deliver("d-2", "field-7"))
        assertFalse(take.deliver("d-1", "field-8"))
    }

    @Test
    fun aPhraseCanBeHeldBackWhileItIsStillBeingSpoken() {
        val direct = catalog.mode("direct")
        assertTrue(catalog.couldBecomePhrase("Draft", direct))
        assertTrue(catalog.couldBecomePhrase("message", direct))
        assertTrue(catalog.couldBecomePhrase("liter", direct).not())
        assertFalse(catalog.couldBecomePhrase("draft mode", direct))
        assertFalse(catalog.couldBecomePhrase("hello", direct))
        assertFalse(catalog.couldBecomePhrase("draft", catalog.mode("verbatim")))
    }

    @Test
    fun aPrivateFieldBlocksEveryPhrase() {
        val routed = catalog.route("draft mode hunter2", catalog.mode("direct"), secure = true)
        assertEquals("blocked", routed.status)
        assertEquals("draft mode hunter2", routed.payload)
    }

    @Test
    fun plansFollowTheMode() {
        assertEquals(ModeCatalog.Plan.NONE, catalog.plan(catalog.mode("direct"), powerSaver = false))
        assertEquals(ModeCatalog.Plan.NONE, catalog.plan(catalog.mode("verbatim"), powerSaver = false))
        assertEquals(ModeCatalog.Plan.RULES, catalog.plan(catalog.mode("draft"), powerSaver = true))
        val clean = catalog.mode("draft").copy(transformKinds = listOf("clean"), authoringRoute = "local-authoring-s1")
        assertEquals(ModeCatalog.Plan.RULES_NO_MODEL, catalog.plan(clean, powerSaver = false))
        assertEquals(ModeCatalog.Plan.RULES_POWER_SAVER, catalog.plan(clean, powerSaver = true))
    }
}
