package dev.starling.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import dev.starling.mobile.engine.OnDeviceEngine

/**
 * Keeps a model download going with Starling out of sight: a `dataSync`
 * foreground service, so Android neither freezes nor kills the process
 * while the app is in the background or its task was swiped away, plus a
 * partial wake lock so the transfer goes on with the screen off. The
 * download itself stays in [ModelDownloadController]; this service only
 * follows its state, shows the progress notification (whose Pause stops the
 * download, keeping the partial file for a resume), and leaves once the
 * download is no longer running, with a notification saying how it ended.
 * Everything here runs on the main thread.
 */
class ModelDownloadService : Service() {
    private val downloads by lazy { starlingApplication().modelDownloads }
    private var wakeLock: PowerManager.WakeLock? = null
    private var listening = false
    private var lastProgressPost = 0L

    // The newest start command: stopping with it leaves a service that a
    // newer download already started again running.
    private var lastStartId = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Every start command counts, the Pause action's too: stopping with an
        // older id would leave the service running.
        lastStartId = startId
        if (intent?.action == ACTION_PAUSE) {
            downloads.cancel()
            // Nothing running: no state change will come to stop the service.
            if (!downloads.isRunning) finish(null)
            return START_NOT_STICKY
        }
        // startForegroundService() obliges startForeground() even when the
        // download already ended; the service then leaves again at once.
        val started = runCatching {
            ServiceCompat.startForeground(
                this,
                PROGRESS_ID,
                progressNotification(null),
                if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0,
            )
        }.onFailure { Log.w(TAG, "download foreground service refused", it) }.isSuccess
        if (!started) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (!listening) {
            listening = true
            // Delivers the current state at once.
            downloads.addListener(listener)
        }
        return START_NOT_STICKY
    }

    private val listener: (ModelDownloadController.State) -> Unit = { state ->
        when (state) {
            is ModelDownloadController.State.Running -> {
                holdWakeLock()
                // Progress arrives every MiB; the notification needs far less.
                val now = SystemClock.elapsedRealtime()
                if (state.verifying || now - lastProgressPost >= PROGRESS_INTERVAL_MS) {
                    lastProgressPost = now
                    notificationManager()?.notify(PROGRESS_ID, progressNotification(state))
                }
            }
            else -> finish(state)
        }
    }

    /**
     * Android 15+ ends a `dataSync` service after its daily budget: pause,
     * keeping the partial file, and say so (a user pause says nothing).
     */
    override fun onTimeout(startId: Int, fgsType: Int) {
        downloads.cancel()
        finish(null)
        notificationManager()?.notify(
            RESULT_ID,
            baseNotification().setContentText(getString(R.string.download_notification_timeout, label())).setAutoCancel(true).build(),
        )
    }

    override fun onDestroy() {
        if (listening) downloads.removeListener(listener)
        listening = false
        releaseWakeLock()
        super.onDestroy()
    }

    private fun finish(state: ModelDownloadController.State?) {
        if (listening) downloads.removeListener(listener)
        listening = false
        releaseWakeLock()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf(lastStartId)
        val text = when (state) {
            is ModelDownloadController.State.Finished -> when (val result = state.result) {
                is OnDeviceEngine.ImportResult.Imported -> getString(R.string.download_notification_done, label())
                is OnDeviceEngine.ImportResult.Rejected -> result.reason
            }
            is ModelDownloadController.State.Failed -> getString(R.string.on_device_download_failed, state.reason)
            // Paused by the user, or nothing was running: nothing to report.
            else -> null
        } ?: return
        notificationManager()?.notify(RESULT_ID, baseNotification().setContentText(text).setAutoCancel(true).build())
    }

    private fun progressNotification(state: ModelDownloadController.State.Running?): Notification {
        val builder = baseNotification()
            .setContentTitle(getString(R.string.download_notification_title, label()))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .addAction(
                R.drawable.ic_mic,
                getString(R.string.cancel_download),
                PendingIntent.getService(
                    this,
                    0,
                    Intent(this, ModelDownloadService::class.java).setAction(ACTION_PAUSE),
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                ),
            )
        when {
            state == null -> builder.setProgress(0, 0, true)
            state.verifying -> builder.setProgress(0, 0, true).setContentText(getString(R.string.on_device_verifying))
            else -> {
                val percent = if (state.total > 0) (state.bytes * 100 / state.total).toInt() else 0
                builder.setProgress(100, percent, false)
                    .setContentText(getString(R.string.download_notification_progress, state.bytes / MB, state.total / MB))
            }
        }
        return builder.build()
    }

    private fun baseNotification() = NotificationCompat.Builder(this, ensureChannel(this))
        .setSmallIcon(R.drawable.ic_mic)
        .setContentTitle(getString(R.string.app_name))
        .setContentIntent(
            PendingIntent.getActivity(
                this,
                0,
                Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            ),
        )

    private fun label(): String = downloads.spec?.label ?: getString(R.string.download_notification_model)

    private fun notificationManager() = getSystemService(NotificationManager::class.java)

    private fun holdWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = getSystemService(PowerManager::class.java)
            ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "starling:model-download")
            ?.apply {
                // Bounds a leaked hold; the next progress tick of a longer
                // download takes it again.
                setReferenceCounted(false)
                acquire(WAKE_LOCK_TIMEOUT_MS)
            }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { runCatching { if (it.isHeld) it.release() } }
        wakeLock = null
    }

    companion object {
        private const val TAG = "ModelDownloadService"
        private const val PROGRESS_ID = 7302
        private const val RESULT_ID = 7303
        private const val CHANNEL_ID = "model-download"
        private const val ACTION_PAUSE = "dev.starling.mobile.action.PAUSE_DOWNLOAD"
        private const val PROGRESS_INTERVAL_MS = 1_000L
        private const val WAKE_LOCK_TIMEOUT_MS = 2L * 60 * 60 * 1000
        private const val MB = 1_000_000L

        /**
         * Follows the download that just started. A refusal (Android does
         * not allow a foreground service right now) is logged; the download
         * still runs, only without surviving the app going to the background.
         */
        fun start(context: Context) {
            runCatching {
                ContextCompat.startForegroundService(context, Intent(context, ModelDownloadService::class.java))
            }.onFailure { Log.w(TAG, "download foreground service not started", it) }
        }

        private fun ensureChannel(context: Context): String {
            context.getSystemService(NotificationManager::class.java)?.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    context.getString(R.string.download_notification_channel),
                    NotificationManager.IMPORTANCE_LOW,
                ),
            )
            return CHANNEL_ID
        }
    }
}
