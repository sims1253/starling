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
//!   the user typed. A closed segment remembers its last words and each
//!   partial finds them again, so when the server revises earlier words
//!   (and their count), the words the user saw before the edit stay
//!   before it.
//! - The final transcript closes every segment ([`LiveSegmenter::finish`]).
//!
//! Segment texts carry their own leading space, so the draft's text reads
//! like the transcript: the words joined by single spaces.
//!
//! Word indices follow the server contract: ASCII whitespace separates
//! words, not language-specific tokenization. Unspaced CJK/Thai text is
//! therefore one live segment until stable or final. Editing inside it
//! still pins the entire segment; the editor does not guess subword cuts.

use crate::staging::Draft;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Span {
    segment: u32,
    start: usize,
    /// `None` for the open segment: it takes every word after `start`.
    end: Option<usize>,
    /// A closed segment's last words (see [`tail_of`]), to find its end again.
    tail: String,
}

/// How many trailing words anchor a closed segment's end.
const ANCHOR_WORDS: usize = 3;
/// How far (in words) a revision may move an anchor.
const ANCHOR_REACH: usize = 12;

/// A word as anchors compare it: lowercase letters and digits only, so a
/// revised comma or capital does not lose the anchor.
fn norm(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// A closed segment's anchor: its last words as one run of letters, so a
/// revision that splits or merges them ("anymore" → "any more") still
/// matches.
fn tail_of(words: &[&str], start: usize, end: usize) -> String {
    let end = end.min(words.len());
    let from = end.saturating_sub(ANCHOR_WORDS).max(start.min(end));
    words[from..end].iter().map(|word| norm(word)).collect()
}

/// Whether the words of `words[start..end]` end with the letters `tail`
/// (the match must end exactly at `end`).
fn ends_with_tail(words: &[&str], start: usize, end: usize, tail: &str) -> bool {
    let mut letters = String::new();
    for word in words[start..end].iter().rev() {
        letters.insert_str(0, &norm(word));
        if letters.len() >= tail.len() {
            break;
        }
    }
    letters.ends_with(tail)
        && words[..end]
            .last()
            .is_some_and(|word| !norm(word).is_empty())
}

/// Finds the nearest surviving anchor, searching nearby first. A large
/// server revision may move the anchor beyond the usual overlap window.
fn realign(words: &[&str], tail: &str, start: usize, expected: usize) -> Option<usize> {
    let start = start.min(words.len());
    let expected = expected.clamp(start, words.len());
    if tail.is_empty() {
        return Some(expected);
    }
    let low = (start + 1).max(expected.saturating_sub(ANCHOR_REACH));
    let high = words.len().min(expected.saturating_add(ANCHOR_REACH));
    let nearest = |range: std::ops::RangeInclusive<usize>| {
        range
            .filter(|&end| ends_with_tail(words, start, end, tail))
            .min_by_key(|&end| end.abs_diff(expected))
    };
    nearest(low..=high).or_else(|| nearest(start + 1..=words.len()))
}

/// An exact deletion can erase a closed segment's entire anchor. Map its
/// old boundary through that deletion instead of consuming later speech.
fn deleted_boundary(old: &[String], new: &[&str], boundary: usize) -> Option<usize> {
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    (prefix + suffix == new.len() && (prefix..=old.len() - suffix).contains(&boundary))
        .then_some(prefix)
}

#[derive(Debug, Clone, Default)]
pub struct LiveSegmenter {
    /// Closed segments not final yet, in word order.
    closed: Vec<Span>,
    open: Option<Span>,
    next_segment: u32,
    /// The words already final, as they were when finalized.
    finalized: Vec<String>,
    /// The latest partial's words.
    heard: Vec<String>,
    /// A partial or the final disagreed with words already final.
    diverged: bool,
    finished: bool,
}

/// The draft text of words `[start, end)`: joined by single spaces, with a
/// leading space unless the span starts the take.
fn span_text(words: &[&str], start: usize, end: usize, segment: u32) -> String {
    let start = start.min(words.len());
    let end = end.clamp(start, words.len());
    let joined = words[start..end].join(" ");
    if (start > 0 || segment > 0) && !joined.is_empty() {
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
        match self.closed.last() {
            Some(Span { end: Some(end), .. }) => *end,
            _ => self.finalized.len(),
        }
    }

    fn open_span(&mut self) -> Span {
        let start = self.open_start();
        let span = self.open.get_or_insert_with(|| {
            let span = Span {
                segment: self.next_segment,
                start,
                end: None,
                tail: String::new(),
            };
            self.next_segment += 1;
            span
        });
        // Follows the closed segment before it, which may have moved.
        span.start = start;
        span.clone()
    }

    /// Re-finds every closed segment's end in `words`, oldest first; a
    /// move shifts the expectation for the ones after it.
    fn realign_closed(&mut self, words: &[&str]) {
        let mut start = self.finalized.len().min(words.len());
        let mut shift: isize = 0;
        for span in &mut self.closed {
            let old_end = span.end.unwrap_or(start);
            let expected = old_end.saturating_add_signed(shift);
            let end = deleted_boundary(&self.heard, words, old_end)
                .filter(|end| *end >= start)
                .or_else(|| realign(words, &span.tail, start, expected))
                .or_else(|| {
                    // A revision may change the first anchor word while
                    // its final two words still identify the boundary.
                    let old: Vec<&str> = self.heard.iter().map(String::as_str).collect();
                    let end = old_end.min(old.len());
                    let from = end.saturating_sub(2).max(span.start.min(end));
                    let tail = tail_of(&old, from, end);
                    (!tail.is_empty())
                        .then(|| realign(words, &tail, start, expected))
                        .flatten()
                })
                .unwrap_or_else(|| {
                    // A replacement erased the anchor: retain the best
                    // boundary, but report that its placement is uncertain.
                    self.diverged = true;
                    expected.clamp(start, words.len())
                });
            shift = end as isize - old_end as isize;
            span.start = start;
            span.end = Some(end);
            span.tail = tail_of(words, start, end);
            start = end;
        }
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
        let text = span_text(words, span.start, end, span.segment);
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
        let words: Vec<&str> = text.split_ascii_whitespace().collect();
        self.check_prefix(&words);
        let stable = stable_words.min(words.len());
        self.realign_closed(&words);
        self.heard = words.iter().map(|word| word.to_string()).collect();

        // Closed segments, oldest first: final once stable covers them.
        let mut still_closed = Vec::with_capacity(self.closed.len());
        for span in std::mem::take(&mut self.closed) {
            let end = span.end.unwrap_or(words.len());
            if end == span.start {
                // An exactly deleted segment has no words left to stabilize.
                // Close it without freezing any earlier, still-live words.
                draft.final_attempt(span.segment, &attempt_id(span.segment), "");
                continue;
            }
            if still_closed.is_empty() && stable >= end {
                self.finalize(draft, &words, span, end);
            } else {
                // An empty update would drop the region; a later partial
                // would then re-create it at the end, out of place.
                let text = span_text(&words, span.start, end, span.segment);
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
            let text = span_text(&words, span.start, words.len(), span.segment);
            if !text.is_empty() {
                draft.partial(span.segment, &text);
            }
        } else if let Some(span) = self.open.as_ref() {
            // The entire unedited tail disappeared from this partial.
            draft.partial(span.segment, "");
        }
    }

    /// The user edited the draft: the open segment ends at the words heard
    /// so far, and later speech starts a new segment after the edit.
    pub fn cut(&mut self) {
        if self.finished {
            return;
        }
        if let Some(span) = self.open.take() {
            let heard = self.heard.len();
            if heard > span.start {
                let words: Vec<&str> = self.heard.iter().map(String::as_str).collect();
                self.closed.push(Span {
                    end: Some(heard),
                    tail: tail_of(&words, span.start, heard),
                    ..span
                });
            } else {
                self.open = Some(span);
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
        let words: Vec<&str> = text.split_ascii_whitespace().collect();
        self.check_prefix(&words);
        self.realign_closed(&words);
        if self.open.is_some() {
            self.open_span();
        }
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
    fn native_stable_word_counts_do_not_split_non_ascii_spaces() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "hello\u{a0}world next", 1);
        assert_eq!(draft.raw_text(), "hello\u{a0}world");
        assert_eq!(draft.text(), "hello\u{a0}world next");
    }

    #[test]
    fn a_deleted_segment_after_an_unstable_segment_disappears_immediately() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "first words", 0);
        live.cut();
        draft.insert(draft.text().chars().count(), " [one]");
        live.partial(&mut draft, "first words mistaken words", 0);
        live.cut();
        draft.insert(draft.text().chars().count(), " [two]");
        live.partial(&mut draft, "first words mistaken words later speech", 0);
        live.partial(&mut draft, "first words later speech", 0);
        assert_eq!(draft.text(), "first words [one] [two] later speech");
        assert_eq!(draft.raw_text(), "");
    }

    #[test]
    fn a_disappearing_open_tail_clears_its_partial() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "mistake", 0);
        live.partial(&mut draft, "", 0);
        assert_eq!(draft.text(), "");
        live.partial(&mut draft, "correct", 0);
        assert_eq!(draft.text(), "correct");
    }

    #[test]
    fn a_deleted_closed_segment_does_not_swallow_later_speech() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "mistaken words", 0);
        live.cut();
        draft.insert(draft.text().chars().count(), " [note]");
        live.partial(&mut draft, "mistaken words later speech", 0);
        live.partial(&mut draft, "later speech", 0);
        assert_eq!(draft.text(), " [note] later speech");
        live.finish(&mut draft, "later speech");
        assert_eq!(draft.text(), " [note] later speech");
    }

    #[test]
    fn large_revisions_keep_the_anchor_before_the_edit() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "the old anchor", 0);
        live.cut();
        draft.insert(draft.text().chars().count(), " [note]");
        let prefix = "new ".repeat(20);
        live.partial(&mut draft, &format!("{prefix}the old anchor later"), 0);
        assert_eq!(draft.text(), format!("{prefix}the old anchor [note] later"));
    }

    #[test]
    fn a_shorter_final_after_a_closed_segment_does_not_panic() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "one two three four", 2);
        live.cut();
        draft.insert(draft.text().chars().count(), " [note]");
        assert!(!live.finish(&mut draft, "one"));
        assert!(draft.text().contains("[note]"));
    }

    #[test]
    fn editing_an_unsegmented_script_still_pins_the_users_text() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "你好世界", 0);
        live.cut();
        draft.delete(2, 4);
        draft.insert(2, "朋友");
        live.partial(&mut draft, "你好世界今天", 0);
        live.finish(&mut draft, "你好世界今天");
        assert_eq!(draft.text(), "你好朋友");
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
        assert!(
            kinds(&draft)
                .iter()
                .all(|(kind, _)| *kind == RegionKind::Raw)
        );
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
    fn words_seen_before_an_edit_stay_before_it_when_the_server_revises() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "a b c d e f", 0);
        let end = draft.text().chars().count();
        live.cut();
        draft.insert(end, " [NOTE]");
        // The server inserts two words early on and adds new speech; by
        // index the cut would now fall before "e f".
        live.partial(&mut draft, "a x y b c d e f g h", 0);
        assert_eq!(draft.text(), "a x y b c d e f [NOTE] g h");
        // Or drops a word: the cut must not swallow "g".
        live.partial(&mut draft, "a b c d e f g h i", 0);
        assert_eq!(draft.text(), "a b c d e f [NOTE] g h i");
        // Or merges two of the words the user saw: by index the cut would
        // now take "g" too.
        live.partial(&mut draft, "a b c de f g h i", 0);
        assert_eq!(draft.text(), "a b c de f [NOTE] g h i");
        assert!(live.finish(&mut draft, "a b c de f g h i j"));
        assert_eq!(draft.text(), "a b c de f [NOTE] g h i j");
        assert_eq!(draft.raw_text(), "a b c de f g h i j");
    }

    #[test]
    fn a_word_split_by_a_revision_stays_before_the_edit() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "well I do not wish to see it anymore.", 0);
        let end = draft.text().chars().count();
        live.cut();
        draft.insert(end, " [NOTE]");
        live.partial(
            &mut draft,
            "well I do not wish to see it any more, observed Phoebe",
            0,
        );
        assert_eq!(
            draft.text(),
            "well I do not wish to see it any more, [NOTE] observed Phoebe"
        );
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
