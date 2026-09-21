/**
 * Display formatting shared by the Insights surfaces (E29). Pure string
 * rendering of numbers the metric layer computes — the frozen oracle output
 * keeps raw seconds, and nothing here feeds back into it. One rendering per
 * figure exists so the recap, the share card and the dashboard cannot drift
 * apart on the same number.
 */

/** Captured minutes with one decimal — the single minutes rendering. */
export function formatMinutes(totalMinutes: number): string {
  return `${totalMinutes.toFixed(1)} min`;
}
