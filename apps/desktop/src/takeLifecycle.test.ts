import { describe, expect, it } from "vite-plus/test";

import { TakeLifecycle } from "./takeLifecycle";

describe("TakeLifecycle", () => {
  it("ignores a second start while the first is still starting", () => {
    const lifecycle = new TakeLifecycle();

    expect(lifecycle.beginStart()).toBe(true);
    expect(lifecycle.current()).toBe("starting");

    expect(lifecycle.beginStart()).toBe(false);
    expect(lifecycle.current()).toBe("starting");
  });

  it("ignores a stop while the start transition is still in flight", () => {
    const lifecycle = new TakeLifecycle();

    lifecycle.beginStart();

    expect(lifecycle.beginStop()).toBe(false);
    expect(lifecycle.current()).toBe("starting");
  });

  it("settles a rejected or failed start back to idle so a retry is possible", () => {
    const lifecycle = new TakeLifecycle();

    lifecycle.beginStart();
    lifecycle.endStart(false);

    expect(lifecycle.current()).toBe("idle");
    expect(lifecycle.beginStart()).toBe(true);
  });

  it("ignores a start while the preceding stop is still finalizing", () => {
    const lifecycle = new TakeLifecycle();

    lifecycle.beginStart();
    lifecycle.endStart(true);

    expect(lifecycle.beginStop()).toBe(true);
    expect(lifecycle.current()).toBe("stopping");

    expect(lifecycle.beginStart()).toBe(false);
    expect(lifecycle.current()).toBe("stopping");
  });

  it("ignores a second stop while the first is still finalizing", () => {
    const lifecycle = new TakeLifecycle();

    lifecycle.beginStart();
    lifecycle.endStart(true);
    lifecycle.beginStop();

    expect(lifecycle.beginStop()).toBe(false);
    expect(lifecycle.current()).toBe("stopping");
  });

  it("accepts a new take only after the stop settled back to idle", () => {
    const lifecycle = new TakeLifecycle();

    lifecycle.beginStart();
    lifecycle.endStart(true);
    lifecycle.beginStop();
    lifecycle.endStop();

    expect(lifecycle.current()).toBe("idle");
    expect(lifecycle.beginStart()).toBe(true);
    expect(lifecycle.beginStop()).toBe(false);
    lifecycle.endStart(true);

    expect(lifecycle.beginStop()).toBe(true);
  });
});
