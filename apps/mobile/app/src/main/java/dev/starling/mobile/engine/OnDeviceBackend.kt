package dev.starling.mobile.engine

import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.InferenceResult
import java.io.File

/** Runs transcription through the on-device engine, ignoring connection settings. */
class OnDeviceBackend(private val engine: OnDeviceEngine) {
    fun transcribe(audioFile: File, @Suppress("UNUSED_PARAMETER") config: BackendConfig): InferenceResult =
        engine.transcribe(audioFile)
}
