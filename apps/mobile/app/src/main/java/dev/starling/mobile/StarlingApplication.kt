package dev.starling.mobile

import android.app.Application
import android.content.Context
import dev.starling.mobile.network.BackendSettings
import dev.starling.mobile.network.TranscriptionCoordinator
import dev.starling.mobile.storage.RecordingStore

class StarlingApplication : Application() {
    lateinit var recordings: RecordingStore
        private set
    lateinit var backendSettings: BackendSettings
        private set
    lateinit var transcription: TranscriptionCoordinator
        private set

    override fun onCreate() {
        super.onCreate()
        recordings = RecordingStore(this)
        backendSettings = BackendSettings(this)
        transcription = TranscriptionCoordinator(recordings, backendSettings)
    }
}

fun Context.starlingApplication(): StarlingApplication =
    applicationContext as StarlingApplication
