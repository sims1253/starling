package dev.starling.mobile.processing

/**
 * The keyboard's modes and the deterministic step behind them (#302): the
 * built-in profiles document (`assets/modes/android-profiles.json`, held to
 * the same contract as every other profiles document by
 * `tests/test_staging.py`) plus the spoken-command and spoken-instruction
 * tables shared with the desktop (`assets/contracts/`).
 *
 * Every routing decision goes through the contract's [resolve]; there is no
 * second router here. The phone has no text model yet (#295), so the only
 * processing is the rules step; a mode that asks for model kinds runs rules
 * only and says so.
 */
class ModeCatalog(
    val profiles: ProfilesDocument,
    val commands: SpokenCommands,
    val instructions: SpokenInstructions,
) {
    init {
        profiles.validate()
    }

    val modes: List<Mode> get() = profiles.profiles

    /** The mode with this id, or the default for an id this build no longer ships. */
    fun mode(id: String?): Mode =
        profiles.profiles.firstOrNull { it.id == id }
            ?: profiles.profiles.first { it.id == profiles.defaultProfile }

    /**
     * Routes one text as the start of a take whose mode is [startMode]: a
     * leading mode phrase or the literal escape may pick another mode for
     * this take; a private field blocks every alias.
     */
    fun route(text: String, startMode: Mode, secure: Boolean): RouteResult =
        resolve(
            ProfilesDocument(startMode.id, profiles.profiles, profiles.rules),
            RouteRequest(rawText = text, secureField = secure),
        )

    /**
     * Whether a live partial could still turn into a mode phrase or the
     * literal escape once more words arrive: every word so far matches the
     * start of one, but not all of it yet. Direct mode holds such a partial
     * back from the field, so a phrase never flashes up in the editor.
     */
    fun couldBecomePhrase(partial: String, startMode: Mode): Boolean {
        if (!startMode.allowSpokenOverrides) return false
        val words = partial.trim().split(WHITESPACE).filter(String::isNotEmpty)
            .map { it.trimEnd(':').lowercase() }
        if (words.isEmpty()) return false
        val phrases = profiles.profiles.flatMap { it.aliases } + LITERAL
        return phrases.any { phrase ->
            val target = phrase.trim().lowercase().split(WHITESPACE)
            words.size < target.size && target.subList(0, words.size) == words
        }
    }

    /** What processing a mode gets on this device right now. */
    fun plan(mode: Mode, powerSaver: Boolean): Plan = when {
        mode.behavior == "verbatim" -> Plan.NONE
        mode.transformKinds.isNotEmpty() && powerSaver -> Plan.RULES_POWER_SAVER
        mode.transformKinds.isNotEmpty() -> Plan.RULES_NO_MODEL
        mode.spokenCommands || mode.snippets.isNotEmpty() -> Plan.RULES
        else -> Plan.NONE
    }

    enum class Plan {
        /** The raw text is the output. */
        NONE,

        /** Spoken commands and snippets, no model. */
        RULES,

        /** The mode asks for a model step; battery saver runs rules only. */
        RULES_POWER_SAVER,

        /** The mode asks for a model step this device cannot run yet (#295). */
        RULES_NO_MODEL,
    }

    companion object {
        private const val LITERAL = "literal"
        private val WHITESPACE = Regex("\\s+")
    }
}
