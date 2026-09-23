package dev.starling.mobile.network

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.net.URI
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Derives the `WS /stream` URL from a configured batch endpoint under the
 * same trusted-host rules as the batch client: the endpoint is validated by
 * [EndpointPolicy] first (HTTP only with the explicit opt-in and only for
 * loopback/private/link-local/CGNAT/`.local` hosts), then the scheme is
 * mapped http -> ws and https -> wss. Batch route suffixes are stripped so a
 * reverse-proxied base path still works, but the route itself never applies:
 * `/stream` is a different endpoint. Returns null when the endpoint does not
 * pass the policy, in which case the caller simply records without streaming.
 */
internal fun streamUrl(endpoint: String, allowTrustedLanHttp: Boolean): String? {
    val validation = EndpointPolicy.validate(endpoint, allowTrustedLanHttp)
    if (validation !is EndpointValidation.Valid) return null
    val uri = runCatching { URI(validation.endpoint) }.getOrNull() ?: return null
    val authority = uri.rawAuthority ?: return null
    val scheme = if (uri.scheme?.equals("https", ignoreCase = true) == true) "wss" else "ws"
    val path = streamBasePath(uri.rawPath ?: "")
    return "$scheme://$authority$path/stream"
}

/** Strips a trailing batch route, longest first; keeps any other base path. */
private fun streamBasePath(rawPath: String): String {
    var path = rawPath.trimEnd('/')
    for (route in STREAM_ROUTE_SUFFIXES) {
        if (path.endsWith(route, ignoreCase = true)) {
            path = path.substring(0, path.length - route.length).trimEnd('/')
            break
        }
    }
    return path
}

// Batch route suffixes stripped when deriving the stream URL, longest first
// so /v1/audio/transcriptions strips fully.
private val STREAM_ROUTE_SUFFIXES = listOf(
    "/v1/audio/transcriptions",
    "/audio/transcriptions",
    "/transcriptions",
)

/** One live `WS /stream` dictation session handed to the recording UI. */
interface StreamSession {
    /**
     * Whether audio forwarded to [onAudio] can still reach the server. False
     * once the stream was interrupted, finished, or closed; callers use it to
     * stop forwarding chunks, never to stop recording.
     */
    fun acceptsAudio(): Boolean

    /**
     * Forwards one raw PCM16 16 kHz mono chunk (exactly what AudioCapture
     * writes). Safe to call from the capture worker thread; copies the bytes
     * before returning. Chunks that arrive before the socket opens are
     * buffered (bounded) and flushed in order on open, so the stream's final
     * transcript never silently misses the start of the recording.
     */
    fun onAudio(bytes: ByteArray, count: Int)

    /**
     * Sends `commit` and blocks until the server's `final` message, the
     * configured final timeout, or a failure — whichever comes first. Call on
     * a worker thread, after the WAV has been made durable: [CommitOutcome.
     * Fallback] means the stream failed at any point (connect, mid-stream,
     * or at commit) and the batch upload of the saved WAV must take over.
     * Idempotent; later calls return the first outcome.
     */
    fun finish(): CommitOutcome

    /** Drops the session without committing (recording cancelled or failed). */
    fun close()
}

/**
 * OkHttp WebSocket client for the native server's `WS /stream` dictation
 * protocol (docs/native-serving.md). The capture's PCM16 chunks go out as
 * binary frames; `partial` texts are surfaced as [StreamEvent]s; on
 * [StreamSession.finish] a `commit` is sent and the `final` transcript
 * awaited, with bounded retries when the server answers `server busy`.
 * Every failure mode resolves to [CommitOutcome.Fallback] so the durable WAV
 * batch path can take over; nothing here can fail the recording itself.
 * App-level `ping`/`pong` frames keep the session observable while recording.
 */
class StreamClient(
    private val http: OkHttpClient = defaultHttpClient(),
    private val scheduler: ScheduledExecutorService = defaultScheduler(),
    private val finalTimeoutMillis: Long = FINAL_TIMEOUT_MILLIS,
    private val keepaliveIntervalMillis: Long = KEEPALIVE_INTERVAL_MILLIS,
    private val busyRetryDelaysMillis: LongArray = BUSY_RETRY_DELAYS_MILLIS,
    /**
     * Test seam invoked inside the locked backlog drain after each flushed
     * chunk, so a test can observe the drain in progress and prove that
     * audio arriving meanwhile queues behind the backlog instead of jumping
     * it. A no-op in production.
     */
    private val drainObserver: () -> Unit = {},
) {
    /**
     * Opens a stream to [url] (from [streamUrl]). [events] is invoked on the
     * client's internal threads — including the capture worker thread for
     * interruptions noticed while forwarding audio — so UI callers must
     * marshal to their own thread.
     */
    fun connect(url: String, events: (StreamEvent) -> Unit): StreamSession =
        LiveSession(url, events).also { it.start() }

    private inner class LiveSession(
        private val url: String,
        private val events: (StreamEvent) -> Unit,
    ) : WebSocketListener(), StreamSession {
        private val lock = Any()

        // Guarded by [lock]. `settled` is the single terminal flag: once set,
        // the outcome is fixed, late socket callbacks are ignored, and
        // finish() returns the stored outcome. Every terminal transition
        // goes through [settleLocked], which settles exactly once, releases
        // all finish() waiters, cancels keepalive and pending retries,
        // clears the backlog, and drops the owned socket — so no terminal
        // path can leak the server connection or leave a waiter blocked.
        private var socket: WebSocket? = null
        private var connected = false
        private var closedByClient = false
        private var interruptedReason: String? = null
        private var finishing = false
        private var busyRetries = 0
        private var awaitingPong = false
        private var missedPongs = 0
        private val backlog = ArrayDeque<ByteString>()
        private var backlogBytes = 0L
        private var keepalive: ScheduledFuture<*>? = null
        private var pendingRetry: ScheduledFuture<*>? = null
        private var finishLatch: CountDownLatch? = null
        private val settled = AtomicBoolean(false)

        @Volatile
        private var outcome: CommitOutcome? = null

        fun start() {
            try {
                socket = http.newWebSocket(Request.Builder().url(url).build(), this)
                keepalive = scheduler.scheduleWithFixedDelay(
                    ::keepaliveTick,
                    keepaliveIntervalMillis,
                    keepaliveIntervalMillis,
                    TimeUnit.MILLISECONDS,
                )
            } catch (exception: Exception) {
                fail("the stream could not start: ${exception.message ?: exception.javaClass.simpleName}")
            }
        }

        override fun acceptsAudio(): Boolean = synchronized(lock) {
            !closedByClient && interruptedReason == null && !settled.get()
        }

        override fun onAudio(bytes: ByteArray, count: Int) {
            if (count <= 0) return
            val chunk = bytes.toByteString(0, count)
            var sendFailure: String? = null
            synchronized(lock) {
                if (!acceptsAudioLocked()) return
                if (!connected) {
                    backlogBytes += chunk.size
                    if (backlogBytes > BACKLOG_LIMIT_BYTES) {
                        // The connection never opened; never buffer unbounded
                        // audio. Refusing the stream (not the recording) keeps
                        // the stop-time fallback authoritative.
                        sendFailure = "the stream connection took too long to open"
                    } else {
                        backlog.addLast(chunk)
                    }
                } else {
                    if (!sendLocked(chunk)) {
                        sendFailure = "the stream connection stopped accepting audio"
                    }
                }
            }
            sendFailure?.let(::fail)
        }

        override fun finish(): CommitOutcome {
            val latch = CountDownLatch(1)
            var awaitOn: CountDownLatch? = null
            val early: CommitOutcome? = synchronized(lock) {
                if (!settled.get()) {
                    when {
                        closedByClient -> settleLocked(CommitOutcome.Fallback("the stream was closed before commit"))
                        interruptedReason != null ->
                            settleLocked(CommitOutcome.Fallback(interruptedReason!!))
                        !connected -> {
                            settleLocked(CommitOutcome.Fallback("the stream connection never opened"))
                        }
                        else -> {
                            finishing = true
                            // Concurrent finish() callers await the same
                            // latch: overwriting it would orphan the first
                            // waiter's latch, parking it for the whole final
                            // timeout even after the outcome settled.
                            awaitOn = finishLatch ?: latch.also { finishLatch = it }
                            if (!sendTextLocked(COMMIT_FRAME)) {
                                settleLocked(
                                    CommitOutcome.Fallback("the stream connection stopped accepting the commit"),
                                )
                            }
                        }
                    }
                }
                outcome
            }
            if (early != null) return early
            val completed = try {
                (awaitOn ?: latch).await(finalTimeoutMillis, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                false
            }
            if (!completed) {
                synchronized(lock) {
                    if (!settled.get()) {
                        settleLocked(CommitOutcome.Fallback("the final transcript timed out"))
                    }
                }
            }
            return outcome ?: CommitOutcome.Fallback("the stream ended without a final transcript")
        }

        override fun close() {
            synchronized(lock) {
                if (closedByClient) return
                closedByClient = true
                if (!settled.get()) {
                    // A finish() blocked on the latch must be released
                    // immediately: cancelling the socket alone only unblocks
                    // it once the socket callbacks run, and onClosing/onClosed
                    // route to fail(), which stays silent after this flag.
                    settleLocked(CommitOutcome.Fallback("the stream was closed before commit"))
                }
            }
        }

        // WebSocketListener — OkHttp reader/writer threads.

        override fun onOpen(webSocket: WebSocket, response: Response) {
            var flushFailed = false
            val dead = synchronized(lock) {
                when {
                    settled.get() || closedByClient || interruptedReason != null -> true
                    else -> {
                        connected = true
                        awaitingPong = false
                        missedPongs = 0
                        // Drain the whole backlog under the same lock hold
                        // that made this connection live. Releasing between
                        // chunks would let an onAudio call direct-send a
                        // newer chunk ahead of older backlog chunks, and the
                        // server reassembles PCM in arrival order — a
                        // garbled final transcript with no failure signal.
                        // sendLocked only enqueues, so this costs one lock
                        // acquisition either way.
                        while (backlog.isNotEmpty()) {
                            if (!sendLocked(backlog.removeFirst())) {
                                flushFailed = true
                                break
                            }
                            drainObserver()
                        }
                        backlog.clear()
                        backlogBytes = 0
                        false
                    }
                }
            }
            if (dead) {
                runCatching { webSocket.close(NORMAL_CLOSE, null) }
                return
            }
            if (flushFailed) {
                fail("the stream connection stopped accepting audio")
                return
            }
            events(StreamEvent.Live)
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            when (val message = StreamMessage.parse(text)) {
                is StreamMessage.Partial -> {
                    val deliver = synchronized(lock) { interruptedReason == null && !settled.get() }
                    if (deliver) events(StreamEvent.Partial(message.text))
                }
                is StreamMessage.Final -> {
                    synchronized(lock) {
                        if (finishing && !settled.get()) {
                            settleLocked(CommitOutcome.Final(message.text))
                        }
                    }
                }
                is StreamMessage.Error -> {
                    val retryBusy = synchronized(lock) {
                        finishing && !settled.get() &&
                            message.message == StreamMessage.SERVER_BUSY_MESSAGE
                    }
                    if (retryBusy) retryCommit() else fail(message.message, message.bufferLimitReached)
                }
                StreamMessage.Pong -> synchronized(lock) {
                    awaitingPong = false
                    missedPongs = 0
                }
                StreamMessage.ResetAck, null -> Unit
            }
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            runCatching { webSocket.close(NORMAL_CLOSE, null) }
            fail("the server closed the stream connection")
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            fail("the server closed the stream connection")
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            fail("the stream connection was lost: ${t.message ?: t.javaClass.simpleName}")
        }

        // Internals. Methods taking [lock] must not be called with it held.

        private fun acceptsAudioLocked(): Boolean =
            !closedByClient && interruptedReason == null && !settled.get()

        private fun sendLocked(payload: ByteString): Boolean =
            runCatching { socket?.send(payload) }.getOrDefault(false) == true

        /**
         * Control frames must go out as WebSocket TEXT frames: a JSON frame
         * sent as binary would never reach the server's message parser.
         */
        private fun sendTextLocked(payload: String): Boolean =
            runCatching { socket?.send(payload) }.getOrDefault(false) == true

        /**
         * Marks the stream unusable. Always safe to call from any thread and
         * any state: a session that already settled or was closed stays
         * silent, an interrupted session keeps its first reason, and a
         * finish() waiting for its outcome is released with Fallback
         * immediately instead of burning the whole timeout. Every failure is
         * terminal for the socket whether a commit is pending or not, so it
         * funnels through [settleLocked]: the server frees its session right
         * away and nothing here can outlive the recording.
         */
        private fun fail(reason: String, bufferLimitReached: Boolean = false) {
            val notify: Boolean
            synchronized(lock) {
                if (settled.get() || closedByClient) return
                notify = interruptedReason == null
                if (notify) {
                    interruptedReason = reason
                }
                connected = false
                settleLocked(CommitOutcome.Fallback(interruptedReason!!))
            }
            if (notify) events(StreamEvent.Interrupted(reason, bufferLimitReached))
        }

        /** Retries commit after the documented bounded delay on server busy. */
        private fun retryCommit() {
            var exhausted = false
            synchronized(lock) {
                if (!finishing || settled.get()) return
                val index = busyRetries
                busyRetries++
                if (index >= busyRetryDelaysMillis.size) {
                    exhausted = true
                } else {
                    // Scheduled while holding the lock so a concurrent
                    // settle cancels this exact future: a retry queued
                    // outside the lock could outlive the session and send a
                    // commit on a dropped socket.
                    pendingRetry?.let { pending -> runCatching { pending.cancel(false) } }
                    pendingRetry = scheduler.schedule({
                        synchronized(lock) {
                            pendingRetry = null
                            if (finishing && !settled.get() && connected) {
                                if (!sendTextLocked(COMMIT_FRAME)) {
                                    settleLocked(
                                        CommitOutcome.Fallback("the stream connection stopped accepting the commit"),
                                    )
                                }
                            }
                        }
                    }, busyRetryDelaysMillis[index], TimeUnit.MILLISECONDS)
                }
            }
            if (exhausted) {
                fail("the server stayed busy while finalizing the stream")
            }
        }

        private fun keepaliveTick() {
            var failReason: String? = null
            synchronized(lock) {
                when {
                    settled.get() || closedByClient || !connected || interruptedReason != null -> Unit
                    // While a commit is being finalized the server's read
                    // loop may legitimately stop answering pings; declaring
                    // the stream dead there would trigger a redundant batch
                    // upload long before the documented final budget ends.
                    // Keep pinging so a resumed server resets the counter,
                    // but leave miss counting to the recording phase — a
                    // dead server during commit is caught by onFailure or
                    // the final timeout instead.
                    finishing -> sendTextLocked(PING_FRAME)
                    awaitingPong -> {
                        missedPongs++
                        if (missedPongs >= MISSED_PONG_LIMIT) {
                            failReason = "the stream stopped answering keepalive pings"
                        } else {
                            sendTextLocked(PING_FRAME)
                        }
                    }
                    else -> {
                        awaitingPong = true
                        sendTextLocked(PING_FRAME)
                    }
                }
            }
            failReason?.let(::fail)
        }

        /**
         * The single terminal funnel. Exactly once (CAS): fixes the outcome,
         * releases every finish() waiter, cancels the keepalive and any
         * pending busy retry, clears the backlog, and drops the owned socket
         * so no terminal path — final, fallback, close, or timeout — can leak
         * the server connection. The socket is closed and then cancelled:
         * a live server gets the graceful handshake if the writer can flush
         * it, while the cancel guarantees the connection is dropped even
         * when the peer is wedged (e.g. mid-commit) and would never ack.
         */
        private fun settleLocked(result: CommitOutcome) {
            if (!settled.compareAndSet(false, true)) return
            outcome = result
            cancelKeepaliveLocked()
            cancelPendingRetryLocked()
            backlog.clear()
            backlogBytes = 0
            socket?.let { webSocket ->
                runCatching { webSocket.close(NORMAL_CLOSE, "stream session settled") }
                runCatching { webSocket.cancel() }
            }
            socket = null
            finishLatch?.countDown()
        }

        private fun cancelKeepaliveLocked() {
            keepalive?.let { future -> runCatching { future.cancel(false) } }
            keepalive = null
        }

        private fun cancelPendingRetryLocked() {
            pendingRetry?.let { future -> runCatching { future.cancel(false) } }
            pendingRetry = null
        }
    }

    companion object {
        private const val NORMAL_CLOSE = 1000
        private const val COMMIT_FRAME = """{"type":"commit"}"""
        private const val PING_FRAME = """{"type":"ping"}"""

        /**
         * Longest wait for the `final` message after commit, matching the
         * batch upload's read timeout. The live path normally answers in
         * seconds because windows were finalized while recording; long tails
         * are covered by the batch-equivalent budget before falling back.
         */
        private val FINAL_TIMEOUT_MILLIS = TimeUnit.MINUTES.toMillis(10)

        /** App-level keepalive cadence while the session is live. */
        private val KEEPALIVE_INTERVAL_MILLIS = TimeUnit.SECONDS.toMillis(25)

        /** Consecutive unanswered pings before the stream is presumed dead. */
        private const val MISSED_PONG_LIMIT = 2

        /** Commit retry delays after a `server busy` error frame. */
        private val BUSY_RETRY_DELAYS_MILLIS = longArrayOf(1_000, 3_000)

        /**
         * Cap on audio buffered while waiting for the socket to open
         * (~2 minutes of PCM16). Beyond it the stream is refused early
         * rather than buffering unbounded audio in memory.
         */
        private val BACKLOG_LIMIT_BYTES = 4L * 1024 * 1024

        private fun defaultHttpClient(): OkHttpClient =
            OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                // The WebSocket itself has no read timeout: app-level pings
                // detect liveness, and a wedged socket fails on write.
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .writeTimeout(15, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build()

        /**
         * Single-thread daemon scheduler for keepalives and busy retries.
         * Daemon so it never blocks process exit, and idle between sessions
         * because every keepalive task is cancelled when its session settles.
         */
        private fun defaultScheduler(): ScheduledExecutorService =
            ScheduledThreadPoolExecutor(1) { runnable ->
                Thread(runnable, "starling-stream-keepalive").apply { isDaemon = true }
            }
    }
}
