package dev.starling.mobile

import org.json.JSONObject

/**
 * The licenses of what the app ships, as the build writes them to
 * assets/licenses/third-party-notices.json (app/build.gradle.kts; the same
 * schema as the desktop app's notices).
 */
class ThirdPartyNotices private constructor(
    /** Starling's own license text. */
    val starlingLicense: String,
    /** Sorted by name. */
    val components: List<Component>,
) {
    class Component(
        val name: String,
        val version: String,
        /** An SPDX expression, e.g. "Apache-2.0". */
        val license: String,
        val url: String,
        /** The full license texts, in the order the component lists them. */
        val texts: List<String>,
    )

    companion object {
        const val ASSET = "licenses/third-party-notices.json"

        fun parse(json: String): ThirdPartyNotices {
            val root = JSONObject(json)
            val texts = root.getJSONObject("texts")
            fun text(key: String): String? = texts.optString(key).takeIf { it.isNotEmpty() }
            val components = root.getJSONArray("components").let { array ->
                (0 until array.length()).map { i ->
                    val c = array.getJSONObject(i)
                    val keys = c.getJSONArray("texts")
                    Component(
                        name = c.getString("name"),
                        version = c.optString("version"),
                        license = c.getString("license"),
                        url = c.optString("url"),
                        texts = (0 until keys.length()).mapNotNull { text(keys.getString(it)) },
                    )
                }
            }
            val starling = text(root.getJSONObject("starling").getString("text"))
                ?: throw IllegalArgumentException("third-party notices without Starling's own license")
            return ThirdPartyNotices(starling, components)
        }
    }
}
