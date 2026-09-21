import { useMemo, useState } from "react";
import { Predicate } from "effect";
import { CalendarDays, CircleAlert, Download, RotateCcw, X } from "lucide-react";
import { TRANSFORMATION_KINDS, type InsightEvent } from "./insightEvents";
import {
  aggregate,
  activityByDay,
  localDayKey,
  parseInsightTimestamp,
  selectionStats,
  type InsightAggregate,
} from "./insightMetrics";
import type { InsightConsent } from "./insightConsent";
import type { InsightTermRecord } from "./insightTerms";
import { voicePanel, type VoicePatternCard } from "./insightVoice";
import {
  DEFAULT_SHARE_INCLUDES,
  SHARE_CARD_FIELDS,
  buildShareCard,
  renderShareCard,
  type ShareCardField,
} from "./insightShare";
import { milestonePanel } from "./insightMilestones";
import { weeklyRecap } from "./insightRecap";

/**
 * The Insights surface (E29): Usage and Quality panels computed from the
 * local E28 event log, plus the phase-2 surfaces — the consented "Your
 * voice" cards, milestones/goals/streaks, a matched-period weekly recap and
 * a locally rendered, redacted-by-default share card. The discipline is
 * fixed by INSIGHTS.md and is not decoration: speech words stay separate
 * from generated output, a submission never reads as a confirmed delivery,
 * change counts are never labeled corrected errors, the typing-time
 * comparison states its assumptions and displays negative results, voice
 * cards claim only what their counts support, the share card omits
 * content-derived fields unless explicitly included and is never posted
 * anywhere by the app itself, and anything the data cannot compute renders
 * as "not enough data" — never a fabricated zero. No accuracy, personality
 * or productivity score appears anywhere, because none is knowable.
 */

/** localStorage key for the user's typing baseline (words per minute). */
const TYPING_BASELINE_KEY = "starling:insights:typingWpm";

/** localStorage key for the user's excluded voice-card labels. */
const EXCLUSIONS_KEY = "starling:insights:exclusions";

const CALENDAR_DAYS = 28;

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export interface InsightsViewProps {
  readonly events: readonly InsightEvent[];
  /** The consented content-derived aggregates, when any grant is on. */
  readonly termRecords: readonly InsightTermRecord[];
  readonly consent: InsightConsent;
  /** Applies a whole consent object at once; grants never half-land. */
  readonly onConsentChange: (next: InsightConsent) => void;
  /** Storage/recording issue notice; dismissed by the user. */
  readonly issue?: string;
  readonly onDismissIssue: () => void;
  readonly onReset: () => void;
}

function messageFrom(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

/** A computed panel: either its value, or the reason it could not compute. */
type Computed<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

function compute<T>(work: () => T): Computed<T> {
  try {
    return { ok: true, value: work() };
  } catch (caught) {
    return { ok: false, reason: messageFrom(caught) };
  }
}

function formatMinutes(seconds: number): string {
  return `${(seconds / 60).toFixed(1)} min`;
}

function formatSeconds(seconds: number): string {
  const abs = Math.abs(seconds);
  const minutes = Math.floor(abs / 60);
  const rest = Math.round(abs - minutes * 60);
  const body = `${minutes}:${String(rest).padStart(2, "0")}`;
  const sign = seconds < 0 ? "−" : "";

  return `${sign}${body}`;
}

function plural(count: number, singularWord: string, pluralWord = `${singularWord}s`) {
  return `${count} ${count === 1 ? singularWord : pluralWord}`;
}

function readExclusions(): ReadonlySet<string> {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(EXCLUSIONS_KEY) ?? "[]");

    return new Set(Array.isArray(parsed) ? parsed.filter(Predicate.isString) : []);
  } catch {
    return new Set();
  }
}

/** The one honest placeholder: absence of data, stated as absence. */
function NotEnoughData({ note }: { readonly note: string }) {
  return (
    <div className="insight-tile missing">
      <strong>Not enough data yet</strong>
      <span>{note}</span>
    </div>
  );
}

export function InsightsView({
  events,
  termRecords,
  consent,
  onConsentChange,
  issue,
  onDismissIssue,
  onReset,
}: InsightsViewProps) {
  const timezone = useMemo(() => Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC", []);

  const [baselineDraft, setBaselineDraft] = useState(
    () => localStorage.getItem(TYPING_BASELINE_KEY) ?? "",
  );

  const [excluded, setExcluded] = useState<ReadonlySet<string>>(readExclusions);

  const [shareIncludes, setShareIncludes] = useState<ReadonlySet<ShareCardField>>(
    () => new Set(DEFAULT_SHARE_INCLUDES),
  );

  const [shareCopied, setShareCopied] = useState(false);

  const parsedBaseline = Number.parseInt(baselineDraft, 10);
  const baseline = Number.isInteger(parsedBaseline) && parsedBaseline > 0 ? parsedBaseline : null;

  const usage = useMemo(() => compute(() => aggregate(events, baseline)), [events, baseline]);
  const days = useMemo(() => compute(() => activityByDay(events, timezone)), [events, timezone]);
  const quality = useMemo(() => compute(() => selectionStats(events)), [events]);

  const week = useMemo(() => {
    const sinceMs = new Date().getTime() - WEEK_MS;

    const within = events.filter((event) => parseInsightTimestamp(event.occurred_at) >= sinceMs);

    return compute(() => aggregate(within, baseline));
  }, [events, baseline]);

  const voice = useMemo(
    () => compute(() => voicePanel(events, termRecords, consent, { exclude: excluded })),
    [events, termRecords, consent, excluded],
  );

  const celebrated = useMemo(
    () => compute(() => milestonePanel(events, consent, { timezone })),
    [events, consent, timezone],
  );

  const recap = useMemo(() => compute(() => weeklyRecap(events, { timezone })), [events, timezone]);

  function changeBaseline(value: string) {
    setBaselineDraft(value);
    localStorage.setItem(TYPING_BASELINE_KEY, value);
  }

  function excludeLabel(label: string) {
    const next = new Set(excluded);

    next.add(label);
    setExcluded(next);
    localStorage.setItem(EXCLUSIONS_KEY, JSON.stringify([...next]));
  }

  function toggleConsent(patch: Partial<InsightConsent>) {
    onConsentChange({ ...consent, ...patch });
  }

  function toggleShareField(field: ShareCardField, included: boolean) {
    const next = new Set(shareIncludes);

    if (included) next.add(field);
    else next.delete(field);

    setShareIncludes(next);
    setShareCopied(false);
  }

  function exportAggregate() {
    if (!usage.ok) return;

    const url = URL.createObjectURL(
      new Blob([JSON.stringify(usage.value, null, 2)], { type: "application/json" }),
    );

    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = `starling-insights-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  }

  /**
   * The share card, previewed before anything exists to copy or save. The
   * builder itself enforces the redaction default, so this assembly cannot
   * leak a field the include set does not name.
   */
  const shareCard = useMemo(() => {
    if (!week.ok || week.value.unique_takes === 0) return undefined;

    const milestones = celebrated.ok ? celebrated.value.milestones : [];
    const latestMilestone = milestones.length > 0 ? milestones[milestones.length - 1] : undefined;
    const topPhrase = voice.ok ? voice.value.phraseCards[0] : undefined;

    return buildShareCard(
      {
        rangeStartDay: localDayKey(new Date().getTime() - WEEK_MS, timezone),
        rangeEndDay: localDayKey(new Date().getTime(), timezone),
        words: week.value.recognized_words,
        minutes: week.value.captured_seconds / 60,
        takes: week.value.unique_takes,
        milestone: latestMilestone?.label,
        topPhrase,
      },
      { include: shareIncludes },
    );
  }, [week, celebrated, voice, shareIncludes, timezone]);

  async function copyShareCard() {
    if (shareCard === undefined || !navigator.clipboard) return;

    await navigator.clipboard.writeText(renderShareCard(shareCard));
    setShareCopied(true);
    window.setTimeout(() => setShareCopied(false), 2_000);
  }

  /**
   * Saving is a local download of the previewed text — the only export
   * paths are copy and save; sharing to any site is never automatic.
   */
  function saveShareCard() {
    if (shareCard === undefined) return;

    const url = URL.createObjectURL(new Blob([renderShareCard(shareCard)], { type: "text/plain" }));

    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = `starling-share-card-${new Date().toISOString().slice(0, 10)}.txt`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  }

  function resetInsights() {
    if (
      window.confirm(
        "Reset Insights? Every local insight event is deleted; the dashboard starts over from your next take. Recordings and transcripts are not touched.",
      )
    ) {
      onReset();
    }
  }

  const calendarDays = useMemo(() => {
    const today = new Date();
    const list: Array<{ key: string; label: string }> = [];

    for (let offset = CALENDAR_DAYS - 1; offset >= 0; offset -= 1) {
      const ms = today.getTime() - offset * 24 * 60 * 60 * 1000;
      const key = localDayKey(ms, timezone);

      list.push({
        key,
        label: new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" }),
      });
    }

    return list;
  }, [timezone]);

  const empty = usage.ok && usage.value.unique_takes === 0;

  return (
    <div className="insights-shell">
      <header className="insights-head">
        <div>
          <p className="eyebrow">INSIGHTS</p>
          <h2>What voice made possible</h2>
          <span className="insights-meta">
            Computed locally · formula v1 · reporting timezone {timezone} ·{" "}
            {plural(events.length, "event")}
          </span>
        </div>
        <div className="insights-actions">
          <button onClick={exportAggregate} disabled={!usage.ok}>
            <Download size={15} /> Export aggregate
          </button>
          <button className="danger" onClick={resetInsights}>
            <RotateCcw size={15} /> Reset
          </button>
        </div>
      </header>

      {issue && (
        <div className="insights-issue" role="alert">
          <CircleAlert size={16} />
          <span>{issue}</span>
          <button onClick={onDismissIssue} aria-label="Dismiss insights notice">
            <X size={14} />
          </button>
        </div>
      )}

      {!usage.ok && (
        <div className="insights-failure" role="alert">
          <CircleAlert size={18} />
          <div>
            <strong>Insights could not aggregate the local event log.</strong>
            <span>
              {usage.reason} The recordings themselves are untouched. Reset Insights to start a
              fresh event log, or keep dictating — a conflicting event is a bug worth reporting, not
              a number to hide.
            </span>
          </div>
        </div>
      )}

      {empty && (
        <section className="insights-empty" aria-label="No insights yet">
          <p className="eyebrow">NOTHING HERE YET</p>
          <h3>Your first numbers appear after your first take</h3>
          <p>
            Record a short test dictation from the main screen — a sentence is enough. Insights
            count only what the app can observe, on this machine, with networking off if you prefer.
          </p>
          <div className="insights-example-card" aria-label="Example card, not your data">
            <small>EXAMPLE — NOT YOUR DATA</small>
            <span>312 words from speech · 4.2 minutes captured · 6 takes</span>
          </div>
        </section>
      )}

      {usage.ok && !empty && (
        <section className="insights-panel" aria-label="Usage">
          <div className="insights-panel-head">
            <div>
              <p className="eyebrow">YOUR USAGE</p>
              <h3>Words this app heard you say</h3>
            </div>
            <CalendarDays size={18} />
          </div>

          <div className="insight-tiles">
            <div className="insight-tile">
              <strong>{plural(usage.value.recognized_words, "word")}</strong>
              <span>recognized from speech — final selected transcripts only</span>
            </div>
            <div className="insight-tile">
              <strong>{formatMinutes(usage.value.captured_seconds)}</strong>
              <span>captured audio, silence included</span>
            </div>
            <div className="insight-tile">
              <strong>{plural(usage.value.selected_recognitions, "completed take")}</strong>
              <span>
                takes with a final transcript of {plural(usage.value.unique_takes, "take")} recorded
              </span>
            </div>
            {usage.value.recognized_words_per_captured_minute !== null ? (
              <div className="insight-tile">
                <strong>
                  {Math.round(usage.value.recognized_words_per_captured_minute)} words/min
                </strong>
                <span>
                  recognized words per captured minute — a weighted total, not a speaking-speed
                  truth
                </span>
              </div>
            ) : usage.value.tokenizers.length > 1 ? (
              <NotEnoughData note="words were counted under more than one tokenizer; a shared rate would mislead" />
            ) : (
              <NotEnoughData note="a rate needs a take with a selected transcript and captured audio" />
            )}
          </div>

          <div className="insights-recap">
            <p className="eyebrow">WEEKLY RECAP</p>
            <p className="recap-text">{recapText(recap)}</p>
          </div>

          {celebrated.ok && celebrated.value.enabled && (
            <div className="insights-milestones">
              <p className="eyebrow">MILESTONES — OPTIONAL, YOURS TO SWITCH OFF</p>
              {celebrated.value.milestones.length === 0 ? (
                <NotEnoughData note="milestones appear as totals cross their thresholds — the first take already earns one" />
              ) : (
                <ul>
                  {celebrated.value.milestones.map((milestone) => (
                    <li key={milestone.id}>
                      <strong>{milestone.label}</strong>
                      <span>
                        {milestone.achievedOnDay} — {milestone.evidence}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {celebrated.value.goal !== null && (
                <p className="goal-line">
                  Goal: {celebrated.value.goal.description}
                  {celebrated.value.goal.achieved ? " — reached this week." : "."} A quieter week is
                  never a failure here.
                </p>
              )}
              {celebrated.value.streak !== null && (
                <p className="streak-line">
                  {celebrated.value.streak.description}. Rest days are not failures — the counter
                  just waits.
                </p>
              )}
            </div>
          )}

          <div className="insights-calendar">
            <p className="eyebrow">ACTIVITY · LAST {CALENDAR_DAYS} DAYS</p>
            {days.ok ? (
              <table className="calendar-table">
                <caption className="visually-hidden">
                  Takes and captured minutes per local day for the last {CALENDAR_DAYS} days
                </caption>
                <tbody>
                  {calendarWeeks(calendarDays, days.value).map((weekRow, index) => (
                    <tr key={index}>
                      {weekRow.map((day) => (
                        <td
                          key={day.key}
                          className={day.intensityClass}
                          aria-label={
                            day.activity === undefined
                              ? `${day.label}: no takes — a rest day, not a failure`
                              : `${day.label}: ${plural(day.activity.takes, "take")}, ${formatMinutes(day.activity.captured_seconds)}`
                          }
                        >
                          <span aria-hidden="true">{day.label}</span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <NotEnoughData note={days.reason} />
            )}
            <small>Rest days are not failures; the calendar counts takes, not goals.</small>
          </div>

          <div className="insights-proxy">
            <div className="proxy-head">
              <div>
                <p className="eyebrow">TYPING-TIME COMPARISON</p>
                <span>An estimate with stated assumptions — never a measured gain</span>
              </div>
              <label>
                Your typing speed (WPM)
                <input
                  inputMode="numeric"
                  value={baselineDraft}
                  onChange={(event) => changeBaseline(event.target.value)}
                  placeholder="not set"
                  aria-describedby="proxy-assumptions"
                />
              </label>
            </div>
            <p className="proxy-value" aria-live="polite">
              {proxyText(usage.value, baseline, baselineDraft)}
            </p>
            <ul id="proxy-assumptions" className="proxy-assumptions">
              <li>{usage.value.time_comparison_caveat}</li>
              <li>
                estimate = recognized speech words × 60 / your WPM − captured seconds − measured
                post-Stop wait; unknown waits or an unset baseline mean "not enough data", not zero
              </li>
              <li>
                negative results are shown as-is: if the waits outweighed the benefit, it says so
              </li>
            </ul>
          </div>
        </section>
      )}

      {voice.ok && usage.ok && !empty && (
        <section className="insights-panel" aria-label="Your voice">
          <div className="insights-panel-head">
            <div>
              <p className="eyebrow">YOUR VOICE — OPT-IN, COMPUTED FROM YOUR TRANSCRIPTS</p>
              <h3>Patterns from your own words</h3>
            </div>
          </div>

          <div className="insights-consent">
            <p>
              These cards are the only Insights that read transcript contents, so each kind needs
              its own grant. Aggregates are computed locally, retained per these settings, and
              deleted with the take they came from. Turning a kind off deletes what it had retained.
            </p>
            <label>
              <input
                type="checkbox"
                checked={consent.recurringPhrases}
                onChange={(event) => toggleConsent({ recurringPhrases: event.target.checked })}
              />
              Recurring phrases
            </label>
            <label>
              <input
                type="checkbox"
                checked={consent.vocabularyPatterns}
                onChange={(event) => toggleConsent({ vocabularyPatterns: event.target.checked })}
              />
              Vocabulary patterns
            </label>
            <label>
              <input
                type="checkbox"
                checked={consent.milestones}
                onChange={(event) => toggleConsent({ milestones: event.target.checked })}
              />
              Milestones and goals
            </label>
            <label>
              <input
                type="checkbox"
                checked={consent.streaks}
                onChange={(event) => toggleConsent({ streaks: event.target.checked })}
              />
              Streak counter
            </label>
            <label>
              Weekly word goal (optional)
              <input
                inputMode="numeric"
                value={consent.weeklyGoalWords ?? ""}
                onChange={(event) => {
                  const parsed = Number.parseInt(event.target.value, 10);

                  toggleConsent({
                    weeklyGoalWords: Number.isInteger(parsed) && parsed > 0 ? parsed : null,
                  });
                }}
                placeholder="no goal"
              />
            </label>
          </div>

          {!consent.recurringPhrases && !consent.vocabularyPatterns ? (
            <NotEnoughData note="no content-derived analysis is enabled — enable one above; nothing is read or retained while both are off" />
          ) : (
            <div className="insight-tiles">
              {consent.recurringPhrases &&
                (voice.value.phraseCards.length > 0 ? (
                  voice.value.phraseCards.map((card) => (
                    <VoiceCardTile key={`p-${card.label}`} card={card} onExclude={excludeLabel} />
                  ))
                ) : (
                  <NotEnoughData note="no phrase repeated across takes in the window yet" />
                ))}
              {consent.vocabularyPatterns &&
                (voice.value.vocabularyCards.length > 0 ? (
                  voice.value.vocabularyCards.map((card) => (
                    <VoiceCardTile key={`v-${card.label}`} card={card} onExclude={excludeLabel} />
                  ))
                ) : (
                  <NotEnoughData note="no term repeated across takes in the window yet" />
                ))}
            </div>
          )}
          <small>
            Cards state only what their counts support; excluded labels are kept in local settings
            and never shown again. No card infers traits, moods or ability.
          </small>
        </section>
      )}

      {shareCard !== undefined && (
        <section className="insights-panel" aria-label="Share card">
          <div className="insights-panel-head">
            <div>
              <p className="eyebrow">SHARE CARD — PREVIEWED, REDACTED BY DEFAULT</p>
              <h3>Review exactly what would leave your hands</h3>
            </div>
          </div>

          <pre className="share-card-preview">{renderShareCard(shareCard)}</pre>

          <div className="share-card-fields">
            {SHARE_CARD_FIELDS.map((field) => (
              <label key={field}>
                <input
                  type="checkbox"
                  checked={shareIncludes.has(field)}
                  disabled={
                    field === "topPhrase" && voice.ok && voice.value.phraseCards.length === 0
                  }
                  onChange={(event) => toggleShareField(field, event.target.checked)}
                />
                {field === "topPhrase" ? "top phrase (content-derived)" : field}
              </label>
            ))}
          </div>

          <div className="insights-actions">
            <button onClick={() => void copyShareCard()} disabled={!navigator.clipboard}>
              {shareCopied ? "Copied" : "Copy card text"}
            </button>
            <button onClick={saveShareCard}>Save as text</button>
          </div>
          <small>
            Copy and save are the only actions — Starling never posts anywhere for you. Aggregate
            fields only, unless you explicitly include the content-derived one.
          </small>
        </section>
      )}

      {quality.ok && usage.ok && !empty && (
        <section className="insights-panel" aria-label="Quality">
          <div className="insights-panel-head">
            <div>
              <p className="eyebrow">QUALITY</p>
              <h3>What happened while capturing</h3>
            </div>
          </div>

          <div className="insight-tiles">
            <div className="insight-tile">
              <strong>{plural(quality.value.supersededSelections, "retry")}</strong>
              <span>
                recognitions a newer attempt replaced — a retry revises, it never adds a take
              </span>
            </div>
            <div className="insight-tile">
              <strong>{plural(quality.value.emptySelections, "empty result")}</strong>
              <span>
                takes whose final transcript had no words — distinct from a failed attempt
              </span>
            </div>
            <div className="insight-tile">
              <strong>{plural(usage.value.incomplete_captures, "incomplete capture")}</strong>
              <span>audio flagged incomplete at capture time</span>
            </div>
            <div className="insight-tile">
              <strong>
                {usage.value.delivery_counts.confirmed} confirmed /{" "}
                {usage.value.delivery_counts.submitted_unconfirmed} submitted
              </strong>
              <span>
                deliveries — only an observed copy counts as confirmed; an export is unconfirmed
              </span>
            </div>
            <NotEnoughData note="no event kind carries interrupted-stream recoveries yet" />
            <NotEnoughData note="no event kind carries local-vs-remote processing provenance yet" />
          </div>

          {(usage.value.transformation_counts.model_authoring > 0 ||
            usage.value.transformation_counts.snippet_expansion > 0 ||
            usage.value.transformation_counts.user_edit > 0 ||
            usage.value.transformation_counts.dictionary_substitution > 0) && (
            <div className="insights-changes">
              <p className="eyebrow">CHANGES BY TYPE — NOT CERTIFIED CORRECTED ERRORS</p>
              <table className="changes-table">
                <caption className="visually-hidden">
                  Transformation passes, and word changes between raw and refined transcripts by
                  kind
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Kind</th>
                    <th scope="col">Passes</th>
                  </tr>
                </thead>
                <tbody>
                  {TRANSFORMATION_KINDS.flatMap((kind) =>
                    usage.value.transformation_counts[kind] > 0
                      ? [
                          <tr key={kind}>
                            <th scope="row">{kind.replace(/_/g, " ")}</th>
                            <td>{usage.value.transformation_counts[kind]}</td>
                          </tr>,
                        ]
                      : [],
                  )}
                </tbody>
              </table>
              <p className="changes-summary">
                Word changes across all passes: {usage.value.change_counts.style} style (paired
                swaps), {usage.value.change_counts.structural} structural (net size),{" "}
                {usage.value.change_counts.user} user, {usage.value.change_counts.dictionary}{" "}
                dictionary, {usage.value.change_counts.snippet} snippet.
              </p>
              <small>
                A "style" change is a paired word swap between the raw and refined text; a
                "structural" change is a net size difference. Without a reference transcript,
                accuracy is unknown — fewer changes never mean more accuracy, and deleting words is
                never rewarded.
              </small>
            </div>
          )}

          <details className="insights-definitions">
            <summary>How these numbers are defined</summary>
            <ul>
              <li>
                Recognized words: the word-like segment count (UAX#29 via the host ICU, tokenizer id
                uax29-intl-v1) of each take's final selected transcript. A retry replaces the count;
                it never adds another take's worth.
              </li>
              <li>
                Captured minutes: retained audio frames divided by the actual sample rate — silence
                included. A VAD-normalized speech rate is a distinct metric this version does not
                show, and is never silently substituted.
              </li>
              <li>
                Completed takes: captures with a selected final transcript. An empty recognition
                stays distinguishable from a nonempty one; neither implies acoustic accuracy.
              </li>
              <li>
                Delivery: a clipboard copy that resolved is confirmed; a file export is
                submitted-unconfirmed because its completion is not observable; failures are counted
                as failures.
              </li>
              <li>
                Deleting a recording removes its contribution from every number here — the tombstone
                dominates stale replays, so deleted takes cannot reappear — and deletes its
                content-derived term aggregates with it.
              </li>
              <li>
                The weekly recap compares two consecutive seven-day windows of equal instant length,
                so a clock change never shortens one side; word deltas are counts, never praise.
              </li>
              <li>
                Deliberately absent: accuracy, personality, health and productivity scores. The
                events cannot know them, so no tile pretends to.
              </li>
            </ul>
          </details>
        </section>
      )}
    </div>
  );
}

function VoiceCardTile({
  card,
  onExclude,
}: {
  readonly card: VoicePatternCard;
  readonly onExclude: (label: string) => void;
}) {
  return (
    <div className="insight-tile voice-card">
      <strong>{card.label}</strong>
      <span>{card.description}</span>
      <button className="link" onClick={() => onExclude(card.label)}>
        Exclude this {card.kind === "recurring_phrase" ? "phrase" : "term"}
      </button>
    </div>
  );
}

interface CalendarCell {
  readonly key: string;
  readonly label: string;
  readonly intensityClass: string;
  readonly activity: { readonly takes: number; readonly captured_seconds: number } | undefined;
}

function calendarWeeks(
  dayList: ReadonlyArray<Readonly<{ key: string; label: string }>>,
  activity: ReadonlyMap<string, { takes: number; captured_seconds: number }>,
): ReadonlyArray<ReadonlyArray<CalendarCell>> {
  const cells = dayList.map(({ key, label }) => {
    const day = activity.get(key);

    return {
      key,
      label,
      activity: day,
      intensityClass:
        day === undefined
          ? "cal-0"
          : day.takes === 1
            ? "cal-1"
            : day.takes <= 3
              ? "cal-2"
              : "cal-3",
    };
  });

  const weeks: CalendarCell[][] = [];

  for (let index = 0; index < cells.length; index += 7) {
    weeks.push(cells.slice(index, index + 7));
  }

  return weeks;
}

function recapText(recap: Computed<ReturnType<typeof weeklyRecap>>): string {
  if (!recap.ok) {
    return `The weekly recap could not be computed: ${recap.reason}`;
  }

  if (!recap.value.ok) {
    return "Not enough data yet — takes from the last seven days will fill this in.";
  }

  return recap.value.recap.lines.join(" ");
}

function proxyText(
  usage: InsightAggregate,
  baseline: number | null,
  baselineDraft: string,
): string {
  if (baseline === null) {
    return baselineDraft.trim() === ""
      ? "Not enough data yet — set your typing speed to compare against."
      : "Your typing baseline needs to be a positive whole number of words per minute.";
  }

  const proxy = usage.typing_time_comparison_seconds;

  if (proxy === null) {
    if (usage.selected_recognitions === 0) {
      return "Not enough data yet — a comparison needs a take with a selected transcript.";
    }

    if (usage.tokenizers.length > 1) {
      return "Not enough data yet — words were counted under more than one tokenizer.";
    }

    return "Not enough data yet — some takes have no measured post-Stop wait, and the estimate refuses to guess.";
  }

  const sign = proxy >= 0 ? "about " : "";

  return `${sign}${formatSeconds(proxy)} ${proxy >= 0 ? "saved" : "lost to waiting"} versus typing at ${baseline} WPM (estimate, before unobserved correction time).`;
}
