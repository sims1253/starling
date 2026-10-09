package dev.starling.mobile

import android.text.InputType
import android.view.inputmethod.EditorInfo
import dev.starling.mobile.ui.EditorField
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EditorFieldTest {
    private fun field(
        inputType: Int = InputType.TYPE_CLASS_TEXT,
        imeOptions: Int = 0,
        packageName: String? = "com.example.chat",
        fieldId: Int = 42,
    ) = EditorField(packageName, fieldId, null, inputType, imeOptions, "Message")

    @Test
    fun passwordVariationsAreSensitive() {
        listOf(
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD,
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD,
        ).forEach { type -> assertTrue("type $type", field(inputType = type).sensitive) }
    }

    @Test
    fun noPersonalizedLearningIsSensitive() {
        // Incognito browser tabs set this flag on every field.
        assertTrue(field(imeOptions = EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING).sensitive)
    }

    @Test
    fun ordinaryFieldsAreNotSensitive() {
        assertFalse(field().sensitive)
        assertFalse(field(inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS).sensitive)
        // The number-class password variation value means something else in
        // the text class; it must not mark plain number fields.
        assertFalse(field(inputType = InputType.TYPE_CLASS_NUMBER).sensitive)
        assertFalse(field(inputType = InputType.TYPE_CLASS_PHONE).sensitive)
    }

    @Test
    fun typeNullFieldsCannotCompose() {
        assertFalse(field(inputType = InputType.TYPE_NULL).supportsComposing)
        assertTrue(field().supportsComposing)
    }

    @Test
    fun sameFieldNeedsEveryDeclaredAttribute() {
        assertTrue(field().sameFieldAs(field()))
        assertFalse(field().sameFieldAs(field(fieldId = 43)))
        assertFalse(field().sameFieldAs(field(packageName = "com.example.other")))
        assertFalse(field().sameFieldAs(field(imeOptions = EditorInfo.IME_ACTION_SEND)))
        assertFalse(field().sameFieldAs(field(inputType = InputType.TYPE_CLASS_NUMBER)))
    }

    @Test
    fun fieldWithoutPackageIsNeverRecognized() {
        assertFalse(field(packageName = null).sameFieldAs(field(packageName = null)))
    }
}
