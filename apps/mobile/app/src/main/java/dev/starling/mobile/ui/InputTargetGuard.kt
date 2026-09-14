package dev.starling.mobile.ui

/**
 * Generation plus object identity protects an asynchronous transcript from
 * being committed after the editor target has changed.
 */
class InputTargetGuard<T : Any> {
    data class Snapshot<T : Any>(val generation: Long, val target: T)

    private var generation = 0L
    private var currentTarget: T? = null

    fun targetStarted(target: T) {
        generation += 1
        currentTarget = target
    }

    fun targetFinished() {
        generation += 1
        currentTarget = null
    }

    fun capture(): Snapshot<T>? = currentTarget?.let { Snapshot(generation, it) }

    fun isCurrent(snapshot: Snapshot<T>, target: T?): Boolean =
        target != null && snapshot.generation == generation && snapshot.target === target
}
