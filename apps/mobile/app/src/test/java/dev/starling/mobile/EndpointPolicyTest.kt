package dev.starling.mobile

import dev.starling.mobile.network.EndpointPolicy
import dev.starling.mobile.network.EndpointValidation
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
    fun cgnatVpnHostsNeedOptInAndRangeCheck() {
        assertTrue(EndpointPolicy.validate("http://100.101.42.1:8181", false) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("http://100.63.1.20:8181", true) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("http://100.64.0.1:8181", true) is EndpointValidation.Valid)
        assertTrue(EndpointPolicy.validate("http://100.127.255.254:8181", true) is EndpointValidation.Valid)
        assertTrue(EndpointPolicy.validate("http://100.128.1.20:8181", true) is EndpointValidation.Invalid)
    }

    @Test
    fun credentialsAndFragmentsAreRejected() {
        assertTrue(EndpointPolicy.validate("https://user:secret@server.example", false) is EndpointValidation.Invalid)
        assertTrue(EndpointPolicy.validate("https://server.example/v1/audio/transcriptions?token=secret", false) is EndpointValidation.Invalid)
    }

    @Test
    fun exactOpenAiRouteCanBeConfigured() {
        assertTrue(
            EndpointPolicy.validate("https://server.example/v1/audio/transcriptions", false) is EndpointValidation.Valid,
        )
    }

    @Test
    fun routeSelectionUsesTheSingleTranscriptionApi() {
        assertEquals(
            "https://server.example/v1/audio/transcriptions",
            inferenceUrl("https://server.example"),
        )
        assertEquals(
            "https://server.example/v1/audio/transcriptions",
            inferenceUrl("https://server.example/v1"),
        )
        assertEquals(
            "https://server.example/v1/audio/transcriptions",
            inferenceUrl("https://server.example/v1/audio/transcriptions"),
        )
        // Pre-unification full routes are not migrated (no backwards compatibility).
        assertEquals(
            "https://server.example/transcribe/v1/audio/transcriptions",
            inferenceUrl("https://server.example/transcribe"),
        )
    }
}
