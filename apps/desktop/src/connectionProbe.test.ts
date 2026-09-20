import { describe, expect, it } from "vite-plus/test";
import type { ServerHealth } from "@starling/dictation";

import {
  CheckSequencer,
  connectionFailureMessage,
  probeOutcomeFromFailure,
  probeOutcomeFromHealth,
  settingsCalloutView,
} from "./connectionProbe";

function health(overrides: Partial<ServerHealth> = {}): ServerHealth {
  return { status: "ok", ...overrides };
}

describe("CheckSequencer", () => {
  it("lets only the newest check report", () => {
    const sequencer = new CheckSequencer();
    const first = sequencer.begin();

    expect(sequencer.isCurrent(first)).toBe(true);

    const second = sequencer.begin();

    expect(sequencer.isCurrent(first)).toBe(false);
    expect(sequencer.isCurrent(second)).toBe(true);
  });

  it("cancelAll retires every check still in flight", () => {
    const sequencer = new CheckSequencer();
    const first = sequencer.begin();
    const second = sequencer.begin();

    sequencer.cancelAll();

    expect(sequencer.isCurrent(first)).toBe(false);
    expect(sequencer.isCurrent(second)).toBe(false);

    const fresh = sequencer.begin();

    expect(sequencer.isCurrent(fresh)).toBe(true);
  });
});

describe("connectionFailureMessage", () => {
  it("names the endpoint that was actually probed on transport failures", () => {
    const failure = new Error("Could not reach the transcription server at http://committed:8181.");

    expect(connectionFailureMessage("http://draft:9000", failure)).toBe(
      "Could not reach the transcription server at http://draft:9000.",
    );
  });

  it("keeps validation reasons verbatim", () => {
    const failure = new Error("The endpoint must use http or https.");

    expect(connectionFailureMessage("ftp://draft", failure)).toBe(
      "The endpoint must use http or https.",
    );
  });

  it("stringifies causes that are not errors", () => {
    expect(connectionFailureMessage("http://draft", "boom")).toBe("boom");
  });
});

describe("probe outcomes", () => {
  it("maps a healthy server to a ready probe with its model", () => {
    expect(probeOutcomeFromHealth("http://draft:9000", health({ model: "parakeet" }))).toEqual({
      state: "ok",
      endpoint: "http://draft:9000",
      model: "parakeet",
      busy: false,
    });
  });

  it("falls back to a generic model name and flags busy servers and queue depth", () => {
    expect(probeOutcomeFromHealth("http://draft", health())).toEqual({
      state: "ok",
      endpoint: "http://draft",
      model: "server",
      busy: false,
    });
    expect(probeOutcomeFromHealth("http://draft", health({ busy: true }))).toMatchObject({
      busy: true,
    });
    expect(probeOutcomeFromHealth("http://draft", health({ queueDepth: 2 }))).toMatchObject({
      busy: true,
    });
  });

  it("carries the probed endpoint and a shaped message on failure", () => {
    expect(
      probeOutcomeFromFailure(
        "http://draft:9000",
        new Error("Could not reach the transcription server at http://elsewhere."),
      ),
    ).toEqual({
      state: "failed",
      endpoint: "http://draft:9000",
      message: "Could not reach the transcription server at http://draft:9000.",
    });
  });
});

describe("settingsCalloutView", () => {
  it("shows the live committed status while no probe has run", () => {
    expect(settingsCalloutView(undefined, "ready", "http://committed:8181")).toEqual({
      dot: "ready",
      title: "Server connected",
      detail: "http://committed:8181",
    });
    expect(settingsCalloutView(undefined, "offline", "http://committed:8181").title).toBe(
      "Server needs attention",
    );
  });

  it("shows the probe while it is testing", () => {
    expect(
      settingsCalloutView({ state: "testing", endpoint: "http://draft:9000" }, "ready", "http://c"),
    ).toEqual({ dot: "checking", title: "Testing connection…", detail: "http://draft:9000" });
  });

  it("shows a successful probe without touching the live status", () => {
    const view = settingsCalloutView(
      {
        state: "done",
        outcome: { state: "ok", endpoint: "http://draft:9000", model: "parakeet", busy: false },
      },
      "offline",
      "http://committed:8181",
    );

    expect(view).toEqual({
      dot: "ready",
      title: "parakeet responded",
      detail: "http://draft:9000",
    });

    const busy = settingsCalloutView(
      {
        state: "done",
        outcome: { state: "ok", endpoint: "http://draft", model: "parakeet", busy: true },
      },
      "ready",
      "http://committed",
    );

    expect(busy.dot).toBe("busy");
    expect(busy.title).toBe("parakeet is working");
  });

  it("shows a failed probe's message, not the live status", () => {
    const view = settingsCalloutView(
      {
        state: "done",
        outcome: { state: "failed", endpoint: "http://draft:9000", message: "no server there" },
      },
      "ready",
      "http://committed:8181",
    );

    expect(view).toEqual({ dot: "offline", title: "Probe failed", detail: "no server there" });
  });
});
