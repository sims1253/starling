export type TakePhase = "idle" | "starting" | "recording" | "stopping";

/**
 * Serializes the app-level take lifecycle: begin → start → stop → finalize.
 *
 * The phase is plain synchronous state, so a toggle claims its transition
 * before awaiting anything. The record button and the global shortcut both
 * funnel through the same guard, and a toggle issued while a transition is
 * still in flight is ignored instead of interleaving with it (#143).
 */
export class TakeLifecycle {
  private phase: TakePhase = "idle";

  current(): TakePhase {
    return this.phase;
  }

  /** Claim the start transition (idle → starting); false when one is not due. */
  beginStart(): boolean {
    if (this.phase !== "idle") return false;

    this.phase = "starting";

    return true;
  }

  /** Settle a claimed start: recording when the take was accepted, else idle. */
  endStart(accepted: boolean): void {
    if (this.phase !== "starting") return;

    this.phase = accepted ? "recording" : "idle";
  }

  /** Claim the stop transition (recording → stopping); false when recording is not live. */
  beginStop(): boolean {
    if (this.phase !== "recording") return false;

    this.phase = "stopping";

    return true;
  }

  /** Settle a claimed stop once the take is finalized and the recorder released. */
  endStop(): void {
    if (this.phase === "stopping") this.phase = "idle";
  }
}
