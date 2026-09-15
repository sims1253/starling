package dev.starling.mobile.network

import android.content.Context
import java.net.InetAddress
import java.net.URI

data class BackendConfig(
    val endpoint: String,
    val allowTrustedLanHttp: Boolean,
    val protocol: BackendProtocol = BackendProtocol.STARLING,
    val model: String = "parakeet",
    val engine: TranscriptionEngine = TranscriptionEngine.REMOTE,
)

enum class BackendProtocol {
    STARLING,
    OPENAI,
}

enum class TranscriptionEngine {
    REMOTE,
    ON_DEVICE,
}

/** User-editable, non-secret connection settings. No auth token is persisted. */
class BackendSettings(context: Context) {
    private val preferences = context.applicationContext.getSharedPreferences(
        PREFS_NAME,
        Context.MODE_PRIVATE,
    )

    fun load(): BackendConfig = BackendConfig(
        endpoint = preferences.getString(KEY_ENDPOINT, DEFAULT_ENDPOINT) ?: DEFAULT_ENDPOINT,
        allowTrustedLanHttp = preferences.getBoolean(KEY_ALLOW_HTTP, false),
        protocol = preferences.getString(KEY_PROTOCOL, BackendProtocol.STARLING.name)
            ?.let { value -> runCatching { BackendProtocol.valueOf(value) }.getOrDefault(BackendProtocol.STARLING) }
            ?: BackendProtocol.STARLING,
        model = preferences.getString(KEY_MODEL, DEFAULT_MODEL) ?: DEFAULT_MODEL,
        engine = preferences.getString(KEY_ENGINE, TranscriptionEngine.REMOTE.name)
            ?.let { value -> runCatching { TranscriptionEngine.valueOf(value) }.getOrDefault(TranscriptionEngine.REMOTE) }
            ?: TranscriptionEngine.REMOTE,
    )

    fun save(config: BackendConfig) {
        val validated = EndpointPolicy.validate(config.endpoint, config.allowTrustedLanHttp)
        require(validated is EndpointValidation.Valid) { (validated as EndpointValidation.Invalid).message }
        val model = config.model.trim()
        require(config.protocol != BackendProtocol.OPENAI || model.isNotEmpty()) {
            "Enter the model name served by the backend"
        }
        preferences.edit()
            .putString(KEY_ENDPOINT, validated.endpoint)
            .putBoolean(KEY_ALLOW_HTTP, config.allowTrustedLanHttp)
            .putString(KEY_PROTOCOL, config.protocol.name)
            .putString(KEY_MODEL, model)
            .putString(KEY_ENGINE, config.engine.name)
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "backend_settings"
        private const val KEY_ENDPOINT = "endpoint"
        private const val KEY_ALLOW_HTTP = "allow_trusted_lan_http"
        private const val KEY_PROTOCOL = "protocol"
        private const val KEY_MODEL = "model"
        private const val KEY_ENGINE = "engine"
        const val DEFAULT_MODEL = "parakeet"

        // HTTPS is the safe default. Local development can explicitly opt into
        // http://127.0.0.1:8181 or a private LAN endpoint in the UI.
        const val DEFAULT_ENDPOINT = "https://127.0.0.1:8181"
    }
}

sealed interface EndpointValidation {
    data class Valid(val endpoint: String) : EndpointValidation
    data class Invalid(val message: String) : EndpointValidation
}

/**
 * Runtime policy for the editable endpoint. Cleartext is accepted only after
 * an explicit checkbox and only for loopback/private/link-local addresses or
 * a .local mDNS name. Remote HTTP endpoints are rejected.
 */
object EndpointPolicy {
    fun validate(rawEndpoint: String, allowTrustedLanHttp: Boolean): EndpointValidation {
        val value = rawEndpoint.trim().trimEnd('/')
        if (value.isEmpty()) return EndpointValidation.Invalid("Enter a backend URL")

        val uri = runCatching { URI(value) }.getOrNull()
            ?: return EndpointValidation.Invalid("Enter a valid backend URL")
        val scheme = uri.scheme?.lowercase()
        if (scheme != "https" && scheme != "http") {
            return EndpointValidation.Invalid("Backend URL must use HTTPS (or HTTP for trusted LAN)")
        }
        if (uri.rawUserInfo != null) {
            return EndpointValidation.Invalid("Credentials in the URL are not supported")
        }
        val host = uri.host?.trim('[', ']')?.lowercase()
            ?: return EndpointValidation.Invalid("Backend URL must include a host")
        if (uri.query != null || uri.fragment != null) {
            return EndpointValidation.Invalid("Backend URL cannot contain a query or fragment")
        }
        if (uri.port !in -1..65535) {
            return EndpointValidation.Invalid("Backend URL has an invalid port")
        }
        if (scheme == "http") {
            if (!allowTrustedLanHttp) {
                return EndpointValidation.Invalid("Enable trusted LAN HTTP before using an http:// endpoint")
            }
            if (!isTrustedCleartextHost(host)) {
                return EndpointValidation.Invalid(
                    "HTTP is limited to localhost, private LAN, link-local, or .local hosts",
                )
            }
        }
        return EndpointValidation.Valid(value)
    }

    private fun isTrustedCleartextHost(host: String): Boolean {
        if (host == "localhost" || host.endsWith(".local")) return true
        if (isPrivateIpv4(host)) return true
        if (!host.matches(Regex("[0-9a-f:.%]+")) || !host.contains(':')) return false
        return runCatching {
            val address = InetAddress.getByName(host)
            address.isLoopbackAddress || address.isLinkLocalAddress ||
                address.isSiteLocalAddress || address.hostAddress
                    ?.substringBefore('%')
                    ?.startsWith("fc", true) == true || address.hostAddress
                    ?.substringBefore('%')
                    ?.startsWith("fd", true) == true
        }.getOrDefault(false)
    }

    private fun isPrivateIpv4(host: String): Boolean {
        val octets = host.split('.')
        if (octets.size != 4 || octets.any { it.isEmpty() || (it.length > 1 && it.startsWith('0')) }) {
            return false
        }
        val values = octets.map { it.toIntOrNull() ?: return false }
        if (values.any { it !in 0..255 }) return false
        val first = values[0]
        val second = values[1]
        return first == 10 ||
            (first == 172 && second in 16..31) ||
            (first == 192 && second == 168) ||
            (first == 169 && second == 254) ||
            first == 127
    }
}
