package dev.starling.mobile.storage

import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.IOException

internal object Durability {
    private val ON_ANDROID = System.getProperty("java.vm.name") == "Dalvik"

    /**
     * Makes the entries of [directory] (files created or renamed in it)
     * durable; a file's own fsync does not cover its directory entry. Throws
     * when the sync fails. JVM unit tests have no android.system and skip it.
     */
    fun syncDirectory(directory: File) {
        if (!ON_ANDROID) return
        try {
            val fd = Os.open(directory.path, OsConstants.O_RDONLY, 0)
            try {
                Os.fsync(fd)
            } finally {
                Os.close(fd)
            }
        } catch (exception: ErrnoException) {
            throw IOException("Unable to sync ${directory.name}", exception)
        }
    }
}
