package dev.starling.mobile.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamMessagesTest {
    @Test fun parsesPartialWithBounds() {
        assertEquals(
            StreamMessage.Partial("hello", startSeconds = 0.0, endSeconds = 12.5),
            StreamMessage.parse("""{"type":"partial","text":"hello","start_s":0.0,"end_s":12.5}"""),
        )
    }

    @Test fun parsesFinalWithDurationAndSegmentsIgnored() {
        assertEquals(
            StreamMessage.Final("hello world", durationSeconds = 12.5),
            StreamMessage.parse(
                """{"type":"final","text":"hello world","segments":[{"text":"hello world"}],"duration_s":12.5}""",
            ),
        )
    }

    @Test fun parsesPlainError() {
        assertEquals(
            StreamMessage.Error("server busy", bufferLimitReached = false),
            StreamMessage.parse("""{"type":"error","message":"server busy"}"""),
        )
    }

    @Test fun flagsTheLiveBufferCapErrorWhateverTheConfiguredCap() {
        for (seconds in listOf("60", "30", "0")) {
            val message = StreamMessage.parse(
                """{"type":"error","message":"stream buffer limit reached ($seconds s live buffer); audio ignored until reset"}""",
            )
            assertTrue(message is StreamMessage.Error)
            assertTrue((message as StreamMessage.Error).bufferLimitReached)
        }
    }

    @Test fun parsesControlReplies() {
        assertEquals(StreamMessage.Pong, StreamMessage.parse("""{"type":"pong"}"""))
        assertEquals(StreamMessage.ResetAck, StreamMessage.parse("""{"type":"reset_ack"}"""))
    }

    @Test fun emptyTextIsAValidResult() {
        assertEquals(
            StreamMessage.Final("", durationSeconds = 3.0),
            StreamMessage.parse("""{"type":"final","text":"","duration_s":3.0}"""),
        )
        assertEquals(
            StreamMessage.Partial(""),
            StreamMessage.parse("""{"type":"partial","text":""}"""),
        )
    }

    @Test fun malformedOrUnknownFramesAreIgnored() {
        assertNull(StreamMessage.parse("not json"))
        assertNull(StreamMessage.parse("""{"type":"brand-new-thing"}"""))
        assertNull(StreamMessage.parse("""{"text":"no type"}"""))
        assertNull(StreamMessage.parse("""{"type":"final"}"""))
        assertNull(StreamMessage.parse("""{"type":"partial","start_s":1.0}"""))
        assertNull(StreamMessage.parse("""{"type":"error","message":""}"""))
        assertNull(StreamMessage.parse("[]"))
    }

    @Test fun optionalBoundsDefaultToZero() {
        assertEquals(
            StreamMessage.Partial("hi"),
            StreamMessage.parse("""{"type":"partial","text":"hi"}"""),
        )
        assertFalse(StreamMessage.parse("""{"type":"final","text":"x"}""") is StreamMessage.Error)
    }
}
