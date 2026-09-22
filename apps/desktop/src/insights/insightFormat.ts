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
 * day in a different dress. `withYear` adds the year — the shape a share
 * card's range line uses, because a saved plain-text artifact must stay
 * unambiguous when it is re-read a year later. An unparseable key is shown
 * verbatim — the metric contract guarantees keys, so this is belt-and-braces.
 */
export function formatDayKey(dayKey: string, options?: { readonly withYear?: boolean }): string {
  const date = new Date(`${dayKey}T00:00:00`);

  if (Number.isNaN(date.getTime())) return dayKey;

  const parts: Intl.DateTimeFormatOptions = { month: "short", day: "numeric" };

  if (options?.withYear === true) parts.year = "numeric";

  return date.toLocaleDateString([], parts);
}

/** "take"/"takes", "day"/"days" — the word alone, pluralized by its count. */
export function pluralWord(count: number, singular: string): string {
  return `${singular}${count === 1 ? "" : "s"}`;
}

/** The count with its pluralized noun — one wording across every surface. */
export function plural(count: number, singularWord: string, pluralForm = `${singularWord}s`) {
  return `${count} ${count === 1 ? singularWord : pluralForm}`;
}
