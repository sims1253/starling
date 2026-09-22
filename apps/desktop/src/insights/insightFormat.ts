/**
 * Display formatting shared by the Insights surfaces (E29). Pure string
 * rendering of numbers the metric layer computes — the frozen oracle output
 * keeps raw seconds, and nothing here feeds back into it. One rendering per
 * figure exists so the recap, the share card and the dashboard cannot drift
 * apart on the same number.
 */

/**
 * Captured minutes with one decimal — the single minutes rendering. A real
 * but sub-decimal duration is never rounded down to a fabricated "0.0 min":
 * the discipline every Insights surface states forbids a zero where
 * something happened, so anything from one frame up to a tenth of a minute
 * renders as "<0.1 min" instead.
 */
export function formatMinutes(totalMinutes: number): string {
  if (totalMinutes > 0 && totalMinutes < 0.1) return "<0.1 min";

  return `${totalMinutes.toFixed(1)} min`;
}

/**
 * One local-day rendering for every user-facing surface: the metric layer's
 * day keys ("2026-09-22") become the runtime locale's short date, the same
 * shape the calendar and the history rows use, so no surface shows the same
 * day in a different dress. An unparseable key is shown verbatim — the
 * metric contract guarantees keys, so this is belt-and-braces.
 */
export function formatDayKey(dayKey: string): string {
  const date = new Date(`${dayKey}T00:00:00`);

  return Number.isNaN(date.getTime())
    ? dayKey
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
}
