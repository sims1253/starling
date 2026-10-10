package dev.starling.mobile

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat

/**
 * Keeps the keyboard's take recording while its window is hidden: through a
 * screen lock, an app switch, or an editor change. Android delivers silence
 * to a microphone in a background process (and on Android 14+ requires a
 * `microphone` foreground service for it), so the voice keyboard holds this
 * service for exactly as long as its capture runs. The service owns no audio;
 * the capture stays in the keyboard, in the same process.
 *
 * The ongoing notification's Stop action ends the take through the listener
 * the keyboard registered with [hold]. Everything here runs on the main
 * thread, so the plain companion fields need no synchronization.
 */
class CaptureForegroundService : Service() {
    // Whether startForeground() has run for this instance; until it has,
    // stopping would break the startForegroundService() contract, so a
    // release in that window is left to onStartCommand.
    private var inForeground = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_TAKE) {
            // The keyboard's stop releases the service; a stale notification
            // tap with no take behind it must not leave a started service.
            stopListener?.invoke()
            if (!inForeground) stopSelf(startId)
            return START_NOT_STICKY
        }
        // startForegroundService() obliges startForeground() even when the
        // take was released before this command arrived; the service then
        // leaves the foreground again at once.
        val started = runCatching {
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                notification(),
                if (Build.VERSION.SDK_INT >= 30) ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else 0,
            )
        }.onFailure { Log.w(TAG, "microphone foreground service refused", it) }.isSuccess
        inForeground = started
        if (!started || !held) stop()
        return START_NOT_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        running = this
    }

    override fun onDestroy() {
        if (running === this) running = null
        super.onDestroy()
    }

    private fun stop() {
        inForeground = false
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun notification() = NotificationCompat.Builder(this, ensureChannel(this))
        .setSmallIcon(R.drawable.ic_mic)
        .setContentTitle(getString(R.string.take_notification_title))
        .setContentText(getString(R.string.take_notification_text))
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .setCategory(NotificationCompat.CATEGORY_SERVICE)
        .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
        .addAction(
            R.drawable.ic_mic,
            getString(R.string.keyboard_stop),
            PendingIntent.getService(
                this,
                0,
                Intent(this, CaptureForegroundService::class.java).setAction(ACTION_STOP_TAKE),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            ),
        )
        .build()

    companion object {
        private const val TAG = "CaptureForeground"
        private const val NOTIFICATION_ID = 7301
        private const val CHANNEL_ID = "take"
        private const val ACTION_STOP_TAKE = "dev.starling.mobile.action.STOP_TAKE"

        private var held = false
        private var holder = 0L
        private var stopListener: (() -> Unit)? = null
        private var running: CaptureForegroundService? = null

        /**
         * Enters the foreground for a take that just started and returns its
         * hold token. [onStop] runs when the user taps Stop in the
         * notification. A refusal (an Android version or state that does not
         * allow it) is logged and the take records as before, only without
         * surviving a hidden window.
         */
        fun hold(context: Context, onStop: () -> Unit): Long {
            held = true
            holder += 1
            stopListener = onStop
            runCatching {
                ContextCompat.startForegroundService(
                    context,
                    Intent(context, CaptureForegroundService::class.java),
                )
            }.onFailure {
                Log.w(TAG, "microphone foreground service not started", it)
                held = false
                stopListener = null
            }
            return holder
        }

        /**
         * The capture holding [token] has settled; leave the foreground. A
         * stale token (a newer take holds the service now, possibly from
         * another keyboard instance) changes nothing.
         */
        fun release(token: Long) {
            if (token != holder) return
            held = false
            stopListener = null
            running?.takeIf { it.inForeground }?.stop()
        }

        private fun ensureChannel(context: Context): String {
            context.getSystemService(NotificationManager::class.java)?.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    context.getString(R.string.take_notification_channel),
                    NotificationManager.IMPORTANCE_LOW,
                ),
            )
            return CHANNEL_ID
        }
    }
}
