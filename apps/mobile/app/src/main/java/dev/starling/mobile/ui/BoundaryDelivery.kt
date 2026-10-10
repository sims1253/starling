package dev.starling.mobile.ui

import android.view.inputmethod.InputConnection
import dev.starling.mobile.processing.InsertionBoundary

/**
 * One delivery of dictated text into the focused field with the
 * insertion-boundary rules (#341): the leading space and the case of the
 * first letter follow the text around the cursor, read from the field right
 * before the single `commitText`.
 *
 * The surrounding text is read for this decision only; nothing here keeps
 * it, logs it or hands it on. Private fields ([EditorField.sensitive]:
 * passwords, incognito) are never read, and verbatim modes skip the read:
 * both get their text unchanged, as does a field that does not report its
 * text.
 */
object BoundaryDelivery {
    /** Characters read on each side of the cursor; the rules only need the end of the text before it. */
    const val WINDOW = 128

    /** Why the rules were not applied. */
    enum class Skip {
        /** A password or incognito field: its text is not read. */
        PROTECTED,

        /** A verbatim mode: no rule applies. */
        VERBATIM,

        /** The field did not report the text around the cursor. */
        UNREADABLE,
    }

    data class Result(
        /** Whether the editor accepted the commit. */
        val committed: Boolean,
        /** The text handed to `commitText`. */
        val text: String,
        val changes: List<InsertionBoundary.Change>,
        /** Set when the rules could not run; [text] is then the input. */
        val skipped: Skip?,
    )

    /**
     * The text around the cursor, or null when the field does not report it.
     * [composing] is the take's own composing text, with the cursor at its
     * end: the boundary is where that region starts, so it is cut from the
     * text before the cursor. If the field no longer ends there (the cursor
     * moved), the boundary is unknown.
     */
    fun context(connection: InputConnection, field: EditorField, composing: String? = null): InsertionBoundary.Context? {
        if (field.sensitive) return null
        val own = composing.orEmpty()
        val beforeCursor = connection.getTextBeforeCursor(WINDOW + own.length, 0)?.toString() ?: return null
        val after = connection.getTextAfterCursor(WINDOW, 0)?.toString() ?: return null
        if (!beforeCursor.endsWith(own)) return null
        val before = beforeCursor.dropLast(own.length)
        // Some editors report their placeholder as the text of an empty
        // field; one that holds exactly its hint is treated as empty.
        val hint = field.hint
        val whole = beforeCursor.length < WINDOW + own.length && after.length < WINDOW
        val showingHint = own.isEmpty() && !hint.isNullOrEmpty() && whole && before + after == hint
        return InsertionBoundary.Context(before, after, showingHint)
    }

    /**
     * Reads the field, applies the rules to [raw] and commits the result
     * once. [composing] as in [context]; the commit replaces that region.
     */
    fun deliver(
        connection: InputConnection,
        field: EditorField,
        raw: String,
        verbatim: Boolean,
        composing: String? = null,
    ): Result {
        // An empty text has no boundary; nothing is read for it.
        if (raw.isEmpty()) return Result(connection.commitText(raw, 1), raw, emptyList(), null)
        val skipped = when {
            field.sensitive -> Skip.PROTECTED
            verbatim -> Skip.VERBATIM
            else -> null
        }
        val context = if (skipped == null) context(connection, field, composing) else null
        val adjusted = context?.let { InsertionBoundary.adjust(raw, it, verbatim = false) }
            ?: InsertionBoundary.Adjustment(raw, emptyList())
        val committed = connection.commitText(adjusted.text, 1)
        return Result(committed, adjusted.text, adjusted.changes, skipped ?: Skip.UNREADABLE.takeIf { context == null })
    }

    /** The rules applied to [raw] without writing anything, for live composing text. */
    fun adjust(raw: String, context: InsertionBoundary.Context?, verbatim: Boolean): String =
        if (context == null) raw else InsertionBoundary.adjust(raw, context, verbatim).text
}
