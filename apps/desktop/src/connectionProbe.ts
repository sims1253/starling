import type { ServerHealth } from "@starling/dictation";

/**
 * Latest-wins sequencing for asynchronous connection checks (B06). Every
 * check claims a token before awaiting anything and may report its outcome
 * only while its token is still the newest: a slower, older check — against
 * an endpoint or protocol that has since been replaced — resolves after a
 * newer one and is dropped instead of overwriting its result or the live
 * connection status. `cancelAll` retires every in-flight token at once, so
 * closing the settings dialog leaves a stray probe with nothing to land in.
 */
export class CheckSequencer {
  private latest = 0;

  /** Claim the right to report; the returned token is current until the next begin. */
  begin(): number {
    this.latest += 1;

    return this.latest;
  }

  isCurrent(token: number): boolean {
    return token === this.latest;
  }

  cancelAll(): void {
    this.latest += 1;
  }
}

/**
 * The isolated outcome of one Test Connection press (B06). It belongs to the
 * settings dialog alone: the live connection status, server model, and
 * global error state are never written from a probe.
 */
export type ConnectionProbeOutcome =
  | { state: "ok"; endpoint: string; model: string; busy: boolean }
  | { state: "failed"; endpoint: string; message: string };

/** One Test Connection press: in flight, or settled with its own outcome. */
export type ConnectionProbe =
  | { state: "testing"; endpoint: string }
  | { state: "done"; outcome: ConnectionProbeOutcome };

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * One message shape for unreachable servers, shared by the live status and
 * the isolated probe (B06). Transport failures arrive pre-wrapped naming the
 * URL that was tried — for a probe that is the draft endpoint, not the
 * committed one — so the message is re-pointed at the endpoint actually
 * probed. Validation reasons (bad scheme, embedded credentials) arrive
 * verbatim and stay verbatim.
 */
export function connectionFailureMessage(endpoint: string, cause: unknown): string {
  const message = messageFrom(cause);

  return message.startsWith("Could not reach the transcription server")
    ? `Could not reach the transcription server at ${endpoint}.`
    : message;
}

export function probeOutcomeFromHealth(
  endpoint: string,
  health: ServerHealth,
): ConnectionProbeOutcome {
  return {
    state: "ok",
    endpoint,
    model: health.model ?? "server",
    busy: health.busy || (health.queueDepth ?? 0) > 0,
  };
}

export function probeOutcomeFromFailure(endpoint: string, cause: unknown): ConnectionProbeOutcome {
  return { state: "failed", endpoint, message: connectionFailureMessage(endpoint, cause) };
}

/** What the settings dialog's status line shows. */
export interface SettingsCalloutView {
  /** status-dot modifier: ready, busy, checking, or offline. */
  dot: string;

  title: string;

  /** The endpoint probed, or the failure message — never the live endpoint mixed in. */
  detail: string;
}

/**
 * Decide the dialog's status line (B06): a probe that ran owns the line —
 * testing, its own outcome, its own endpoint — and only without one does the
 * line fall back to the live status of the committed endpoint. A failed
 * draft probe therefore never paints the committed connection as offline.
 */
export function settingsCalloutView(
  probe: ConnectionProbe | undefined,
  liveStatus: string,
  liveEndpoint: string,
): SettingsCalloutView {
  if (probe === undefined) {
    return {
      dot: liveStatus,
      title: liveStatus === "ready" ? "Server connected" : "Server needs attention",
      detail: liveEndpoint,
    };
  }

  if (probe.state === "testing") {
    return { dot: "checking", title: "Testing connection…", detail: probe.endpoint };
  }

  if (probe.outcome.state === "failed") {
    return { dot: "offline", title: "Probe failed", detail: probe.outcome.message };
  }

  return {
    dot: probe.outcome.busy ? "busy" : "ready",
    title: probe.outcome.busy
      ? `${probe.outcome.model} is working`
      : `${probe.outcome.model} responded`,
    detail: probe.outcome.endpoint,
  };
}
