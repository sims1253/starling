package dev.starling.mobile.network

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamingEligibilityTest {
    @Test fun remoteServersCanStream() {
        assertTrue(
            TranscriptionCoordinator.streamingEligible(
                BackendConfig(
                    endpoint = "https://server.example:8181",
                    allowTrustedLanHttp = false,
                    engine = TranscriptionEngine.REMOTE,
                ),
            ),
        )
        assertFalse(
            TranscriptionCoordinator.streamingEligible(
                BackendConfig(
                    endpoint = "https://server.example:8181",
                    allowTrustedLanHttp = false,
                    engine = TranscriptionEngine.ON_DEVICE,
                ),
            ),
        )
    }
}
