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

    @Test
    fun aFailedCaptureLeavesTheEarlierDraft() {
        val take = take()
        take.dictate(final = "first comma")
        take.beginSegment()
        take.partial("lost words")
        assertTrue(take.abandonSegment())
        assertFalse(take.recording)
        assertEquals("first,", take.displayText())
        val empty = take()
        empty.beginSegment()
        empty.partial("lost")
        assertFalse(empty.abandonSegment())
    }

    @Test
    fun editsNeverShiftACommandOntoOtherText() {
        // Review #302-5: a cached phrase offset swallowed the payload after an edit.
        val take = take()
        take.dictate(final = "draft mode hello comma world")
        assertEquals("hello, world", take.displayText())
        take.selectWordAt(7)
        take.deleteWord()
        assertEquals("hello,", take.displayText().trimEnd())
        take.dictate("more", final = "more words", attempt = "a-2")
        assertEquals("hello, more words", take.deliveryText().replace("  ", " "))
    }

    @Test
    fun undoingTheNewestInstructionKeepsOlderOnesAside() {
        // Review #302-6.
        val take = take()
        take.dictate(final = "first Starling, make it formal")
        take.dictate(final = "second Starling, make it shorter", attempt = "a-2")
        assertEquals(StagedTake.Decision.Instruction("make it shorter"), take.decision)
        assertEquals("first second", take.deliveryText())
        take.undoDecision()
        assertEquals(StagedTake.Decision.Instruction("make it formal"), take.decision)
        assertEquals("first second Starling, make it shorter", take.deliveryText())
    }

    @Test
    fun aHandPickedModeOutranksTheSpokenPhrase() {
        // Review #302-7: the Send opt-out must hold through the final.
        val take = take("direct")
        take.beginSegment()
        take.partial("message mode see you")
        take.switchMode(catalog.mode("draft"))
        take.final("a-1", "message mode see you")
        assertEquals("draft", take.mode.id)
        assertNull(take.decision)
        assertEquals("see you", take.deliveryText())
    }

    @Test
    fun theNextCaptureShowsItsLiveTailAfterTheProcessedText() {
        // Review #302-11.
        val take = take()
        take.dictate(final = "hello comma world")
        take.beginSegment()
        take.partial("again")
        assertEquals("hello, world again", take.displayText())
    }

    @Test
    fun processedLineBreaksAreDeliveredUnchanged() {
        // Review #302-13.
        val take = take()
        take.dictate(final = "hello new line Starling, make it formal")
        assertEquals("hello\n", take.displayText())
        assertEquals("hello\n", take.deliveryText())
    }

    @Test
    fun aSpokenCorrectionReplacesTheSelectedWord() {
        val take = take()
        take.dictate(final = "meet at sex comma ok")
        assertFalse(take.beginCorrection())
        take.selectWordAt(8)
        assertTrue(take.beginCorrection())
        // Frozen until the correction lands: no other edit, no new target.
        take.selectWordAt(0)
        take.deleteWord()
        assertEquals("meet at sex, ok", take.displayText())
        assertTrue(take.finishCorrection("six"))
        assertEquals("meet at six, ok", take.deliveryText())
        assertFalse(take.finishCorrection("again"))
    }

    @Test
    fun aCorrectionNeverInsertsCommandText() {
        // Review #302 round 2, item 4.
        val take = take()
        take.dictate(final = "meet at sex")
        take.selectWordAt(8)
        assertTrue(take.beginCorrection())
        assertTrue(take.finishCorrection("six Starling, make it formal"))
        assertEquals("meet at six", take.deliveryText())
    }

    @Test
    fun anInstructionRightAfterTheModePhraseStaysOut() {
        // Review #302 round 2, item 3.
        val take = take("direct")
        take.dictate(final = "message mode Starling, make it formal")
        assertEquals("message", take.mode.id)
        assertFalse(take.hasPayload)
        assertEquals("", take.deliveryText().trim())
    }

    @Test
    fun acceptedProcessedTextIsNotProcessedAgain() {
        // Review #302 round 2, item 6: the literal escape survives a continuation.
        val take = take()
        take.dictate(final = "say literal comma")
        assertEquals("say comma", take.displayText())
        take.dictate(final = "again period", attempt = "a-2")
        assertEquals("say comma again.", take.displayText())
    }

    @Test
    fun liveTextStopsBeforeAPossibleInstruction() {
        // Review #302-2: an instruction never reaches the field while spoken.
        val instructions = catalog.instructions
        assertEquals("Send the report".length, instructions.liveCut("Send the report Starling,", "en"))
        assertEquals("Send the report".length, instructions.liveCut("Send the report sterling make it", "en"))
        assertNull(instructions.liveCut("Send the report", "en"))
        assertNull(instructions.liveCut("the word literal starling stays", "en"))
    }

    @Test
    fun commandsAttachAcrossCaptures() {
        // Review #302 round 3, item 1.
        val bullets = take().apply {
            dictate(final = "one")
            dictate(final = "bullet two", attempt = "a-2")
        }
        assertEquals("one\n- two", bullets.displayText())
        val period = take().apply {
            dictate(final = "hello comma")
            dictate(final = "period", attempt = "a-2")
        }
        assertEquals("hello.", period.displayText())
    }

    @Test
    fun aLiteralCorrectionKeepsTheDelimiterAsWords() {
        val take = take()
        take.dictate(final = "meet at sex")
        take.selectWordAt(8)
        assertTrue(take.beginCorrection())
        assertTrue(take.finishCorrection("literal six Starling, ok"))
        assertEquals("meet at six Starling, ok", take.deliveryText())
        // Command words in a correction use the grammar's own escape.
        take.selectWordAt(8)
        assertTrue(take.beginCorrection())
        assertTrue(take.finishCorrection("six literal comma"))
        // The raw view (where the correction was made) shows what was said;
        // the processed proposal applies the escape.
        assertEquals("meet at six literal comma Starling, ok", take.displayText())
        take.toggleView()
        assertEquals("meet at six comma Starling, ok", take.deliveryText())
    }

    @Test
    fun aModeSwitchDuringACorrectionNeverLeavesTheOldProposal() {
        // Review #302 round 3, item 3.
        val take = take()
        take.dictate(final = "hello comma")
        take.toggleView()
        take.selectWordAt(0)
        assertTrue(take.beginCorrection())
        take.switchMode(catalog.mode("direct"))
        assertNull(take.proposal)
        take.cancelCorrection()
        assertNull(take.proposal)
        assertEquals("hello comma", take.deliveryText())
    }

    @Test
    fun backToRawStaysAvailableAfterAnEditTheRulesIgnore() {
        val take = take()
        take.dictate(final = "one two")
        assertFalse(take.canRevertToRaw)
        take.deleteWord()
        assertNull(take.proposal)
        assertTrue(take.canRevertToRaw)
        take.backToRaw()
        assertEquals("one two", take.displayText())
    }

    @Test
    fun aCaptureAfterAProcessedLineBreakKeepsItsRawWordsApart() {
        val take = take()
        take.dictate(final = "hello new line")
        take.dictate(final = "again", attempt = "a-2")
        assertEquals("hello\nagain", take.displayText())
        take.backToRaw()
        assertEquals("hello new line again", take.draft.rawText())
    }

    @Test
    fun aCaptureIntoAnEmptiedDraftGetsNoLeadingSpace() {
        val take = take()
        take.dictate(final = "hello")
        take.deleteWord()
        take.dictate(final = "again", attempt = "a-2")
        assertEquals("again", take.deliveryText())
    }
}
