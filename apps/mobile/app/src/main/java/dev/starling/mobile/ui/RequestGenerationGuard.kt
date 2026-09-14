package dev.starling.mobile.ui

/** Invalidates an older asynchronous result when a newer capture begins. */
class RequestGenerationGuard {
    private var generation = 0L

    fun begin(): Long {
        generation += 1
        return generation
    }

    fun isCurrent(requestGeneration: Long): Boolean = requestGeneration == generation
}
