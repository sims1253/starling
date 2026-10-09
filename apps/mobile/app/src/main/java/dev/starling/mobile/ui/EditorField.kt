package dev.starling.mobile.ui

import android.text.InputType
import android.view.inputmethod.EditorInfo

/**
 * What the keyboard knows about the focused field, taken from its
 * `EditorInfo` and never from its text. It is held in memory only, for two
 * decisions: whether the field is private (see [sensitive]) and whether a
 * field that comes back after a screen lock or an app switch is the one a
 * take started in (see [sameFieldAs]).
 */
data class EditorField(
    val packageName: String?,
    val fieldId: Int,
    val fieldName: String?,
    val inputType: Int,
    val imeOptions: Int,
    val hint: String?,
) {
    /**
     * Password fields and fields that ask for no personalized learning
     * (incognito tabs, private compose screens). A take in such a field
     * leaves no history entry and feeds no dataset.
     */
    val sensitive: Boolean
        get() = isPassword(inputType) || imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING != 0

    /** Whether the field can hold composing text (TYPE_NULL editors cannot). */
    val supportsComposing: Boolean
        get() = inputType and InputType.TYPE_MASK_CLASS != InputType.TYPE_NULL

    /**
     * Whether [other] is this field again. Apps restart input with a new
     * InputConnection after the screen unlocks or the user switches back,
     * so connection identity alone would orphan a take that is still
     * recording. Every attribute the app declared for the field must match,
     * and a field without an app package cannot be recognized.
     */
    fun sameFieldAs(other: EditorField): Boolean = packageName != null && this == other

    companion object {
        fun from(info: EditorInfo?): EditorField? = info?.let {
            EditorField(
                packageName = it.packageName,
                fieldId = it.fieldId,
                fieldName = it.fieldName,
                inputType = it.inputType,
                imeOptions = it.imeOptions,
                hint = it.hintText?.toString(),
            )
        }

        fun isPassword(inputType: Int): Boolean {
            val variation = inputType and InputType.TYPE_MASK_VARIATION
            return when (inputType and InputType.TYPE_MASK_CLASS) {
                InputType.TYPE_CLASS_TEXT ->
                    variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                        variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                        variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD
                InputType.TYPE_CLASS_NUMBER -> variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD
                else -> false
            }
        }
    }
}
