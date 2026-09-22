import { describe, expect, it } from "vite-plus/test";

import { formatDayKey, formatMinutes } from "./insightFormat";

/**
 * Minutes- and day-rendering coverage (E29): one decimal for real durations
 * (never a fabricated "0.0 min" for a capture that happened but lasted less
 * than a tenth of a minute), and one runtime-locale day rendering shared by
 * every user-facing surface — the same never-a-zero discipline the surfaces
 * state.
 */
describe("formatMinutes", () => {
  it("renders whole and fractional minutes with one decimal", () => {
    expect(formatMinutes(0)).toBe("0.0 min");
    expect(formatMinutes(4.2)).toBe("4.2 min");
    expect(formatMinutes(0.1)).toBe("0.1 min");
  });

  it("never rounds a real sub-decimal duration down to a fabricated zero", () => {
    expect(formatMinutes(1 / 60)).toBe("<0.1 min");
    expect(formatMinutes(5.9 / 60)).toBe("<0.1 min");
  });
});

describe("formatDayKey", () => {
  it("renders a day key as the runtime locale's short date", () => {
    // Asserted against the same Intl call, so the test follows the runtime
    // locale instead of pinning one.
    expect(formatDayKey("2026-09-22")).toBe(
      new Date("2026-09-22T00:00:00").toLocaleDateString([], {
        month: "short",
        day: "numeric",
      }),
    );
  });

  it("shows an unparseable key verbatim rather than an invented date", () => {
    expect(formatDayKey("not-a-day")).toBe("not-a-day");
  });
});
