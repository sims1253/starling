package dev.starling.mobile

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import androidx.annotation.RequiresApi
import java.io.File

/**
 * Where installed models live. Once the user allows "All files access"
 * (Android 11+), in Starling/models on shared storage: the folder outlives
 * an uninstall and every Starling build on the phone (release,
 * experimental, debug) uses the same copy, so a reinstall finds its models
 * again. Until then, in app-private storage, which goes with the app. Which
 * model is active stays per install, in [privateDir].
 */
class ModelStorage(private val context: Context) {
    val privateDir = File(context.filesDir, "models")

    @Suppress("DEPRECATION") // The path itself; access comes from MANAGE_EXTERNAL_STORAGE.
    val sharedDir = File(Environment.getExternalStorageDirectory(), "Starling/models")

    /** Whether this Android version offers "All files access" at all. */
    val canShare: Boolean get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R

    /** Whether models are kept in [sharedDir]. Checked per call: the user can grant or revoke it at any time. */
    val isShared: Boolean
        get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && Environment.isExternalStorageManager()

    fun current(): File = if (isShared) sharedDir else privateDir

    /** Models still in [privateDir] while models are kept in [sharedDir]: not moved (yet). */
    fun unmovedModels(): Int =
        if (!isShared) 0 else privateDir.listFiles { file -> file.name.endsWith(".gguf", ignoreCase = true) }?.size ?: 0

    /** Opens this app's "All files access" switch in the system settings. */
    @RequiresApi(Build.VERSION_CODES.R)
    fun grantIntent(): Intent =
        Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${context.packageName}"))
}
