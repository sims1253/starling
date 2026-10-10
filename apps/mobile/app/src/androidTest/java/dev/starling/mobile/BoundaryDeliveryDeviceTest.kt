package dev.starling.mobile

import android.text.InputType
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.widget.EditText
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.ui.BoundaryDelivery
import dev.starling.mobile.ui.EditorField
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #341 against the platform's own editor: a real EditText's InputConnection
 * (what the keyboard is handed for an ordinary text field), so the reads,
 * the composing region and the single commit behave as on a phone, not as
 * the JVM fake assumes.
 */
@RunWith(AndroidJUnit4::class)
class BoundaryDeliveryDeviceTest {
    private class Editor(val view: EditText, val connection: InputConnection, val field: EditorField)

    private fun onMain(block: () -> Unit) = InstrumentationRegistry.getInstrumentation().runOnMainSync(block)

    private fun editor(text: String, inputType: Int = InputType.TYPE_CLASS_TEXT, hint: String? = null): Editor {
        lateinit var editor: Editor
        onMain {
            val view = EditText(ApplicationProvider.getApplicationContext())
            view.inputType = inputType
            view.hint = hint
            view.setText(text)
            view.setSelection(text.length)
            val info = EditorInfo().apply { packageName = view.context.packageName }
            val connection = requireNotNull(view.onCreateInputConnection(info))
            editor = Editor(view, connection, requireNotNull(EditorField.from(info)))
        }
        return editor
    }

    private fun deliver(editor: Editor, raw: String, composing: String? = null): BoundaryDelivery.Result {
        lateinit var result: BoundaryDelivery.Result
        onMain { result = BoundaryDelivery.deliver(editor.connection, editor.field, raw, verbatim = false, composing) }
        return result
    }

    private fun text(editor: Editor): String {
        lateinit var text: String
        onMain { text = editor.view.text.toString() }
        return text
    }

    @Test
    fun midSentenceGetsTheSpaceAndTheLowercaseLetter() {
        val editor = editor("The quick brown")
        val result = deliver(editor, "Fox jumps")
        assertTrue(result.committed)
        assertEquals("The quick brown fox jumps", text(editor))
    }

    @Test
    fun afterASentenceTheCapitalStays() {
        val editor = editor("Done.")
        deliver(editor, "Next one")
        assertEquals("Done. Next one", text(editor))
    }

    @Test
    fun theLiveComposingRegionIsReplacedFromItsOwnStart() {
        val editor = editor("The quick brown")
        onMain { editor.connection.setComposingText(" fox jum", 1) }
        assertEquals("The quick brown fox jum", text(editor))
        deliver(editor, "Fox jumps", composing = " fox jum")
        assertEquals("The quick brown fox jumps", text(editor))
    }

    @Test
    fun anEmptyFieldShowingItsHintStartsTheText() {
        val editor = editor("", hint = "Type a message,")
        // The placeholder is not content: the connection reports nothing.
        onMain {
            val context = requireNotNull(BoundaryDelivery.context(editor.connection, editor.field))
            assertEquals("", context.before)
            assertEquals("", context.after)
        }
        deliver(editor, "Hello")
        assertEquals("Hello", text(editor))
    }

    @Test
    fun aFieldHoldingItsHintsWordsIsRealText() {
        val editor = editor("Message", hint = "Message")
        deliver(editor, "Next")
        assertEquals("Message next", text(editor))
    }

    @Test
    fun aPasswordFieldIsWrittenAsDictated() {
        val editor = editor("secret", inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD)
        assertTrue(editor.field.sensitive)
        val result = deliver(editor, "Word")
        assertEquals(BoundaryDelivery.Skip.PROTECTED, result.skipped)
        assertEquals("secretWord", text(editor))
    }
}
