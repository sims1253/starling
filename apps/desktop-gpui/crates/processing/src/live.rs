//! Live segmentation: a streaming take as draft operations (#297).
//!
//! The `/stream` server sends the whole running transcript with every
//! partial, plus how many of its leading words are stable (no later
//! partial or final can change them). A [`Draft`] wants segments instead:
//! a live partial it may replace while the user leaves it alone, and final
//! attempts that are never rewritten. [`LiveSegmenter`] maps one onto the
//! other by word index:
//!
//! - Stable words become final attempts as soon as the server marks them
//!   stable, so the text the user is most likely to edit is already final.
//! - The words after them are the open segment's partial.
//! - An edit closes the open segment at the words heard so far
//!   ([`LiveSegmenter::cut`]). Those words keep updating in place until
//!   they are final (or the user pins them by editing inside), and later
//!   speech starts a new segment that the draft appends after whatever
//!   the user typed. Words the user already saw never jump past their
//!   edit.
//! - The final transcript closes every segment ([`LiveSegmenter::finish`]).
//!
//! Segment texts carry their own leading space, so the draft's text reads
//! like the transcript: the words joined by single spaces.

use crate::staging::Draft;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    segment: u32,
    start: usize,
    /// `None` for the open segment: it takes every word after `start`.
    end: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct LiveSegmenter {
    /// Closed segments not final yet, in word order.
    closed: Vec<Span>,
    open: Option<Span>,
    next_segment: u32,
    /// The words already final, as they were when finalized.
    finalized: Vec<String>,
    /// How many words the latest partial had.
    heard: usize,
    /// A partial or the final disagreed with words already final.
    diverged: bool,
    finished: bool,
}

/// The draft text of words `[start, end)`: joined by single spaces, with a
/// leading space unless the span starts the take.
fn span_text(words: &[&str], start: usize, end: usize) -> String {
    let start = start.min(words.len());
    let end = end.clamp(start, words.len());
    let joined = words[start..end].join(" ");
    if start > 0 && !joined.is_empty() {
        format!(" {joined}")
    } else {
        joined
    }
}

fn attempt_id(segment: u32) -> String {
    format!("live-{segment}")
}

impl LiveSegmenter {
    pub fn new() -> LiveSegmenter {
        LiveSegmenter::default()
    }

    /// Whether a partial or the final contradicted words already final.
    /// The draft keeps what it showed; the stored transcript is still the
    /// server's final, and "back to raw" on the saved take returns to it.
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Where the open segment starts: after the last closed segment, or
    /// after the final words.
    fn open_start(&self) -> usize {
        match (self.open, self.closed.last()) {
            (Some(span), _) => span.start,
            (None, Some(Span { end: Some(end), .. })) => *end,
            _ => self.finalized.len(),
        }
    }

    fn open_span(&mut self) -> Span {
        let start = self.open_start();
        *self.open.get_or_insert_with(|| {
            let span = Span {
                segment: self.next_segment,
                start,
                end: None,
            };
            self.next_segment += 1;
            span
        })
    }

    fn check_prefix(&mut self, words: &[&str]) {
        let agrees = words.len() >= self.finalized.len()
            && self
                .finalized
                .iter()
                .zip(words)
                .all(|(known, word)| known == word);
        if !agrees {
            self.diverged = true;
        }
    }

    /// Finalizes the segment `span` as words `[span.start, end)`.
    fn finalize(&mut self, draft: &mut Draft, words: &[&str], span: Span, end: usize) {
        let text = span_text(words, span.start, end);
        draft.final_attempt(span.segment, &attempt_id(span.segment), &text);
        let end = end.min(words.len());
        let from = self.finalized.len().min(end);
        self.finalized
            .extend(words[from..end].iter().map(|word| word.to_string()));
    }

    /// One `/stream` partial: the whole running `text` and its stable
    /// word count.
    pub fn partial(&mut self, draft: &mut Draft, text: &str, stable_words: usize) {
        if self.finished {
            return;
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        self.check_prefix(&words);
        self.heard = words.len();
        let stable = stable_words.min(words.len());

        // Closed segments, oldest first: final once stable covers them.
        let mut still_closed = Vec::with_capacity(self.closed.len());
        for span in std::mem::take(&mut self.closed) {
            let end = span.end.unwrap_or(words.len());
            if still_closed.is_empty() && stable >= end {
                self.finalize(draft, &words, span, end);
            } else {
                // An empty update would drop the region; a later partial
                // would then re-create it at the end, out of place.
                let text = span_text(&words, span.start, end);
                if !text.is_empty() {
                    draft.partial(span.segment, &text);
                }
                still_closed.push(span);
            }
        }
        self.closed = still_closed;

        // The open segment: its stable words become final right away, the
        // rest is its partial (a new segment id, since a draft finalizes a
        // segment once).
        if self.closed.is_empty() && stable > self.finalized.len() {
            let span = self.open_span();
            if stable > span.start {
                self.finalize(draft, &words, span, stable);
                self.open = None;
            }
        }
        if words.len() > self.open_start() {
            let span = self.open_span();
            let text = span_text(&words, span.start, words.len());
            if !text.is_empty() {
                draft.partial(span.segment, &text);
            }
        }
    }

    /// The user edited the draft: the open segment ends at the words heard
    /// so far, and later speech starts a new segment after the edit.
    pub fn cut(&mut self) {
        if self.finished {
            return;
        }
        if let Some(span) = self.open {
            if self.heard > span.start {
                self.closed.push(Span {
                    end: Some(self.heard),
                    ..span
                });
                self.open = None;
            }
        }
    }

    /// The final transcript: every segment still open or closed gets its
    /// final attempt. Returns false when the final contradicts words that
    /// were already final (see [`Self::diverged`]).
    pub fn finish(&mut self, draft: &mut Draft, text: &str) -> bool {
        if self.finished {
            return !self.diverged;
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        self.check_prefix(&words);
        let mut spans = std::mem::take(&mut self.closed);
        if spans.is_empty() && self.open.is_none() && words.len() > self.finalized.len() {
            // Nothing live yet (a short take, or no partial arrived).
            self.open_span();
        }
        spans.extend(self.open.take());
        for span in spans {
            let end = span.end.unwrap_or(words.len());
            self.finalize(draft, &words, span, end);
        }
        self.finished = true;
        !self.diverged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staging::RegionKind;

    fn kinds(draft: &Draft) -> Vec<(RegionKind, String)> {
        let snapshot = draft.snapshot();
        snapshot
            .regions
            .iter()
            .map(|region| {
                let text: String = snapshot
                    .text
                    .chars()
                    .skip(region.span[0])
                    .take(region.span[1] - region.span[0])
                    .collect();
                (region.kind, text)
            })
            .collect()
    }

    #[test]
    fn stable_words_become_final_while_the_tail_stays_live() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "one two three", 0);
        assert_eq!(
            kinds(&draft),
            [(RegionKind::Partial, "one two three".into())]
        );
        live.partial(&mut draft, "one two three four five", 2);
        assert_eq!(
            kinds(&draft),
            [
                (RegionKind::Raw, "one two".into()),
                (RegionKind::Partial, " three four five".into()),
            ]
        );
        assert_eq!(draft.raw_text(), "one two");
        assert!(live.finish(&mut draft, "one two three four five six"));
        assert_eq!(draft.text(), "one two three four five six");
        assert_eq!(draft.raw_text(), "one two three four five six");
        assert!(kinds(&draft)
            .iter()
            .all(|(kind, _)| *kind == RegionKind::Raw));
    }

    #[test]
    fn a_revised_tail_replaces_the_partial_in_place() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "we are gonna", 0);
        live.partial(&mut draft, "we are going to go", 0);
        assert_eq!(draft.text(), "we are going to go");
        assert!(live.finish(&mut draft, "we are going to go"));
        assert_eq!(draft.raw_text(), "we are going to go");
    }

    #[test]
    fn typing_at_the_end_keeps_heard_words_before_it_and_new_speech_after() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "hello world and", 0);
        let end = draft.text().chars().count();
        live.cut();
        draft.insert(end, " (note)");
        // The server revises the words already heard and adds more.
        live.partial(&mut draft, "hello world, and more words", 0);
        assert_eq!(draft.text(), "hello world, and (note) more words");
        live.partial(&mut draft, "hello world, and more words here", 4);
        assert_eq!(draft.text(), "hello world, and (note) more words here");
        assert!(live.finish(&mut draft, "hello world, and more words here"));
        assert_eq!(draft.text(), "hello world, and (note) more words here");
        assert_eq!(draft.raw_text(), "hello world, and more words here");
    }

    #[test]
    fn editing_inside_the_live_tail_pins_it_and_speech_continues_after() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "the quick brown fox", 0);
        live.cut();
        // Replace "quick" with "slow" inside the live partial.
        draft.delete(4, 9);
        draft.insert(4, "slow");
        live.partial(&mut draft, "the quack brown fox jumps", 0);
        assert_eq!(draft.text(), "the slow brown fox jumps");
        assert!(live.finish(&mut draft, "the quick brown fox jumps over"));
        // The pinned segment keeps the user's text; its final is recorded
        // as raw, never swapped in.
        assert_eq!(draft.text(), "the slow brown fox jumps over");
        assert_eq!(draft.raw_text(), "the quick brown fox jumps over");
    }

    #[test]
    fn without_stable_words_the_take_is_one_live_segment_until_the_final() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "a b", 0);
        live.partial(&mut draft, "a b c d", 0);
        assert_eq!(kinds(&draft), [(RegionKind::Partial, "a b c d".into())]);
        assert!(live.finish(&mut draft, "a b c d e"));
        assert_eq!(kinds(&draft), [(RegionKind::Raw, "a b c d e".into())]);
    }

    #[test]
    fn a_final_that_contradicts_final_words_is_reported() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "one two three", 2);
        assert!(!live.finish(&mut draft, "uno two three four"));
        assert!(live.diverged());
        // What was final stays; the rest comes from the final by index.
        assert_eq!(draft.text(), "one two three four");
    }

    #[test]
    fn a_take_with_no_partial_is_finalized_whole() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        assert!(live.finish(&mut draft, "short take"));
        assert_eq!(draft.text(), "short take");
        assert_eq!(draft.raw_text(), "short take");
    }

    #[test]
    fn a_closed_segment_waits_for_stable_words_to_cover_it() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "a b c d", 0);
        live.cut();
        live.partial(&mut draft, "a b c d e f", 2);
        // Stable covers only half of the closed segment [0, 4).
        assert_eq!(draft.raw_text(), "");
        live.partial(&mut draft, "a b c d e f g", 5);
        assert_eq!(draft.raw_text(), "a b c d e");
        assert_eq!(draft.text(), "a b c d e f g");
    }
}
