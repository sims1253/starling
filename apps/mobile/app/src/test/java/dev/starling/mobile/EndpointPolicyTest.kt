package dev.starling.mobile

import dev.starling.mobile.network.EndpointPolicy
import dev.starling.mobile.network.EndpointValidation
import dev.starling.mobile.network.BackendProtocol
import dev.starling.mobile.network.inferenceUrl
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test

class EndpointPolicyTest {
    @Test
    fun httpsIsValidByDefault() {
        assertTrue(EndpointPolicy.validate("https://server.example:8181", false) is EndpointValidation.Valid)
    }

    @Test
    fun cleartextNeedsOptInAndPrivateHost() {
        assertTrue(EndpointPolicy.validate("http://192.168.1.20:8181", false) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("http://192.168.1.20:8181", true) is EndpointValidation.Valid)
        assertTrue(EndpointPolicy.validate("http://server.example:8181", true) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("http://[fd00::20]:8181", true) is EndpointValidation.Valid)
    }

    @Test
    fun credentialsAndFragmentsAreRejected() {
        assertTrue(EndpointPolicy.validate("https://user:secret@server.example", false) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("https://server.example/inference?token=secret", false) is EndpointValidation.Invalid)
    }

    @Test
    fun exactOpenAiRouteCanBeConfigured() {
        assertTrue(
            EndpointPolicy.validate("https://server.example/v1/audio/transcriptions", false) is EndpointValidation.Valid,
        )
    }

    @Test
    fun routeSelectionKeepsDocumentedAliasesAndBasePaths() {
        assertEquals(
            "https://server.example/inference",
            inferenceUrl("https://server.example", BackendProtocol.STARLING),
        )
        assertEquals(
            "https://server.example/transcribe",
            inferenceUrl("https://server.example/transcribe", BackendProtocol.STARLING),
        )
        assertEquals(
            "https://server.example/v1/audio/transcriptions",
            inferenceUrl("https://server.example/v1", BackendProtocol.OPENAI),
        )
        assertEquals(
            "https://server.example/v1/audio/transcriptions",
            inferenceUrl("https://server.example/v1/audio/transcriptions", BackendProtocol.OPENAI),
        )
    }
}
