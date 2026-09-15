package dev.starling.mobile

import android.app.Application
import android.content.Context
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.BackendSettings
import dev.starling.mobile.network.TranscriptionCoordinator
import dev.starling.mobile.storage.RecordingStore
import java.io.File

class StarlingApplication : Application() {
    lateinit var recordings: RecordingStore
        private set
    lateinit var backendSettings: BackendSettings
        private set
    lateinit var transcription: TranscriptionCoordinator
        private set
    lateinit var onDeviceEngine: OnDeviceEngine
        private set

    override fun onCreate() {
        super.onCreate()
        recordings = RecordingStore(this)
        backendSettings = BackendSettings(this)
        onDeviceEngine = OnDeviceEngine(File(filesDir, "models"))
        transcription = TranscriptionCoordinator(
            recordings,
            backendSettings,
            OnDeviceBackend(onDeviceEngine),
        )
    }
}

fun Context.starlingApplication(): StarlingApplication =
    applicationContext as StarlingApplication
