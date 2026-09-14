package dev.starling.mobile.network

import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class InferenceClientTest {
    private fun response(status: Int, body: String, check: (InferenceResult, Int) -> Unit) {
        val server = MockWebServer()
        val audio = File.createTempFile("starling-client-", ".wav")
        audio.writeBytes(ByteArray(44))
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                MockResponse().setResponseCode(status).setBody(body)
        }
        server.start()
        try {
            val result = InferenceClient().transcribe(
                audio,
                BackendConfig("http://127.0.0.1:${server.port}", true),
            )
            check(result, server.requestCount)
        } finally {
            server.shutdown()
            audio.delete()
        }
    }

    @Test fun redirectsAreExplicitNonRetryableFailures() {
        for (status in listOf(301, 302, 303, 307, 308)) {
            response(status, "redirect") { result, requests ->
                assertTrue(result is InferenceResult.Failure)
                val failure = result as InferenceResult.Failure
                assertFalse(failure.retryable)
                assertTrue(failure.message.contains("redirect blocked"))
                assertTrue(failure.message.contains(status.toString()))
                assertEquals(1, requests)
            }
        }
    }

    @Test fun serverErrorEnvelopesReachTheUser() {
        for (body in listOf(
            """{"error":{"message":"unknown model"}}""",
            """{"detail":"unknown model"}""",
            """{"error":"unknown model"}""",
            """{"message":"unknown model"}""",
        )) {
            response(400, body) { result, requests ->
                assertEquals(InferenceResult.Failure("unknown model", false), result)
                assertEquals(1, requests)
            }
        }
    }

    @Test fun invalidOrOversizedErrorBodiesKeepTheStatus() {
        for (body in listOf("not JSON", "{}", "x".repeat(2 * 1024 * 1024 + 1))) {
            response(400, body) { result, requests ->
                assertEquals(InferenceResult.Failure("Starling backend returned HTTP 400", false), result)
                assertEquals(1, requests)
            }
        }
    }

    @Test fun successfulTranscriptsKeepTheirWhitespace() {
        response(200, """{"text":"  exact text  "}""") { result, requests ->
            assertEquals(InferenceResult.Success("  exact text  "), result)
            assertEquals(1, requests)
        }
    }

    @Test fun transientErrorsRemainRetryable() {
        response(503, """{"error":{"message":"busy"}}""") { result, requests ->
            assertEquals(InferenceResult.Failure("busy", true), result)
            assertEquals(3, requests)
        }
    }
}
