//! The staging editor (#297): a multi-line, wrapping editor over a staged
//! draft, built like `input.rs` (an entity implementing
//! `EntityInputHandler` plus a custom element) but for paragraphs.
//!
//! The editor does not own the draft. It keeps a display copy of the
//! draft's text, applies the user's edit to that copy at once (the
//! platform input handler reads it back in the same frame), and emits the
//! edit in code points; the app applies it to the draft and pushes the
//! draft's text back with [`StagingEditor::set_content`]. Text that
//! changes without the keyboard (a live partial, an accepted proposal)
//! arrives the same way, and the caret, selection and undo history are
//! remapped around the change.
//!
//! Keyboard editing is first-class (a word or a line is one keystroke):
//! word and line delete, word jumps, visual-line Home/End/Up/Down, and
//! undo/redo, all scoped to the `StagingEditor` key context so they never
//! shadow the app's dictation shortcut.

use std::cell::{Cell, RefCell};
use std::ops::{Deref, Range};
use std::rc::Rc;

use gpui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, Element, ElementId, ElementInputHandler,
    Entity, EntityInputHandler, EventEmitter, FocusHandle, Focusable, FontStyle, GlobalElementId,
    Hsla, IntoElement, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    PaintQuad, Pixels, Point, ScrollHandle, SharedString, Size, Style, TextRun, UTF16Selection,
    UnderlineStyle, Window, WrappedLine, actions, div, fill, point, prelude::*, px, relative, size,
};
use starling_processing::staging::RegionKind;
use unicode_segmentation::UnicodeSegmentation;

use crate::theme;

actions!(
    starling_staging,
    [
        Backspace,
        Delete,
        DeleteWordLeft,
        DeleteWordRight,
        DeleteLine,
        Left,
        Right,
        Up,
        Down,
        WordLeft,
        WordRight,
        SelectLeft,
        SelectRight,
        SelectUp,
        SelectDown,
        SelectWordLeft,
        SelectWordRight,
        Home,
        End,
        SelectHome,
        SelectEnd,
        DocStart,
        DocEnd,
        SelectAll,
        Newline,
        Undo,
        Redo,
        Copy,
        Cut,
        Paste,
        Submit,
        Leave,
    ]
);

const CONTEXT: &str = "StagingEditor";

/// Binds the editor's keys, scoped to its key context.
pub fn bind_keys(cx: &mut App) {
    let editor = Some(CONTEXT);
    cx.bind_keys([
        gpui::KeyBinding::new("backspace", Backspace, editor),
        gpui::KeyBinding::new("shift-backspace", Backspace, editor),
        gpui::KeyBinding::new("delete", Delete, editor),
        gpui::KeyBinding::new("ctrl-backspace", DeleteWordLeft, editor),
        gpui::KeyBinding::new("alt-backspace", DeleteWordLeft, editor),
        gpui::KeyBinding::new("ctrl-delete", DeleteWordRight, editor),
        gpui::KeyBinding::new("alt-delete", DeleteWordRight, editor),
        gpui::KeyBinding::new("secondary-shift-k", DeleteLine, editor),
        gpui::KeyBinding::new("left", Left, editor),
        gpui::KeyBinding::new("right", Right, editor),
        gpui::KeyBinding::new("up", Up, editor),
        gpui::KeyBinding::new("down", Down, editor),
        gpui::KeyBinding::new("ctrl-left", WordLeft, editor),
        gpui::KeyBinding::new("alt-left", WordLeft, editor),
        gpui::KeyBinding::new("ctrl-right", WordRight, editor),
        gpui::KeyBinding::new("alt-right", WordRight, editor),
        gpui::KeyBinding::new("shift-left", SelectLeft, editor),
        gpui::KeyBinding::new("shift-right", SelectRight, editor),
        gpui::KeyBinding::new("shift-up", SelectUp, editor),
        gpui::KeyBinding::new("shift-down", SelectDown, editor),
        gpui::KeyBinding::new("ctrl-shift-left", SelectWordLeft, editor),
        gpui::KeyBinding::new("alt-shift-left", SelectWordLeft, editor),
        gpui::KeyBinding::new("ctrl-shift-right", SelectWordRight, editor),
        gpui::KeyBinding::new("alt-shift-right", SelectWordRight, editor),
        gpui::KeyBinding::new("home", Home, editor),
        gpui::KeyBinding::new("end", End, editor),
        gpui::KeyBinding::new("shift-home", SelectHome, editor),
        gpui::KeyBinding::new("shift-end", SelectEnd, editor),
        gpui::KeyBinding::new("secondary-home", DocStart, editor),
        gpui::KeyBinding::new("secondary-end", DocEnd, editor),
        gpui::KeyBinding::new("secondary-up", DocStart, editor),
        gpui::KeyBinding::new("secondary-down", DocEnd, editor),
        gpui::KeyBinding::new("secondary-a", SelectAll, editor),
        gpui::KeyBinding::new("enter", Newline, editor),
        gpui::KeyBinding::new("secondary-z", Undo, editor),
        gpui::KeyBinding::new("secondary-shift-z", Redo, editor),
        gpui::KeyBinding::new("ctrl-y", Redo, editor),
        gpui::KeyBinding::new("secondary-c", Copy, editor),
        gpui::KeyBinding::new("secondary-x", Cut, editor),
        gpui::KeyBinding::new("secondary-v", Paste, editor),
        gpui::KeyBinding::new("secondary-enter", Submit, editor),
        gpui::KeyBinding::new("escape", Leave, editor),
    ]);
}

/// One edit, in code points (the draft's offsets): replace
/// `[start, end)` with `text`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TextEdit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

pub(crate) enum EditorEvent {
    Edit(TextEdit),
    /// Secondary+Enter: the user is done with the draft.
    Submit,
    /// Escape: hand focus back to the app.
    Leave,
}

// ----------------------------------------------------------------------
// The buffer: text, selection and undo, no UI
// ----------------------------------------------------------------------

/// How an edit joins the undo history: typed characters coalesce into
/// words and repeated deletes into one run, so undo steps feel natural.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Record {
    Typing,
    Deleting,
    Plain,
}

/// At `start` (bytes), `removed` was replaced by `inserted`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct UndoEntry {
    start: usize,
    removed: String,
    inserted: String,
    kind: Record,
}

const UNDO_LIMIT: usize = 500;

/// Where two texts differ: `[prefix, old_end)` in the old text became
/// `[prefix, new_end)` in the new one (bytes, on char boundaries).
fn changed_span(old: &str, new: &str) -> (usize, usize, usize) {
    let prefix: usize = old
        .chars()
        .zip(new.chars())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf8())
        .sum();
    let room = old.len().min(new.len()) - prefix;
    let mut suffix = 0;
    for (a, b) in old[prefix..].chars().rev().zip(new[prefix..].chars().rev()) {
        if a != b || suffix + a.len_utf8() > room {
            break;
        }
        suffix += a.len_utf8();
    }
    (prefix, old.len() - suffix, new.len() - suffix)
}

#[derive(Default)]
pub(crate) struct EditBuffer {
    pub(crate) text: String,
    pub(crate) selected: Range<usize>,
    pub(crate) reversed: bool,
    undo: Vec<UndoEntry>,
    redo: Vec<UndoEntry>,
}

impl EditBuffer {
    pub(crate) fn cursor(&self) -> usize {
        if self.reversed {
            self.selected.start
        } else {
            self.selected.end
        }
    }

    pub(crate) fn move_to(&mut self, offset: usize) {
        let offset = self.floor_boundary(offset);
        self.selected = offset..offset;
        self.reversed = false;
    }

    pub(crate) fn select_to(&mut self, offset: usize) {
        let offset = self.floor_boundary(offset);
        if self.reversed {
            self.selected.start = offset;
        } else {
            self.selected.end = offset;
        }
        if self.selected.end < self.selected.start {
            self.reversed = !self.reversed;
            self.selected = self.selected.end..self.selected.start;
        }
    }

    fn floor_boundary(&self, offset: usize) -> usize {
        let mut offset = offset.min(self.text.len());
        while !self.text.is_char_boundary(offset) {
            offset -= 1;
        }
        offset
    }

    fn code_points(&self, byte: usize) -> usize {
        self.text[..byte].chars().count()
    }

    pub(crate) fn previous_boundary(&self, offset: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .rev()
            .find_map(|(index, _)| (index < offset).then_some(index))
            .unwrap_or(0)
    }

    pub(crate) fn next_boundary(&self, offset: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .find_map(|(index, _)| (index > offset).then_some(index))
            .unwrap_or(self.text.len())
    }

    /// The start of the word before `offset` (whitespace, then a word).
    pub(crate) fn previous_word(&self, offset: usize) -> usize {
        let mut chars = self.text[..offset].char_indices().rev().peekable();
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let mut start = chars.peek().map_or(0, |(index, c)| index + c.len_utf8());
        while let Some((index, _)) = chars.next_if(|(_, c)| !c.is_whitespace()) {
            start = index;
        }
        start
    }

    /// The word (run of non-whitespace) around `offset`.
    pub(crate) fn word_range(&self, offset: usize) -> Range<usize> {
        let offset = self.floor_boundary(offset);
        let start = self.text[..offset]
            .char_indices()
            .rev()
            .take_while(|(_, c)| !c.is_whitespace())
            .last()
            .map_or(offset, |(index, _)| index);
        let end = offset
            + self.text[offset..]
                .char_indices()
                .find(|(_, c)| c.is_whitespace())
                .map_or(self.text.len() - offset, |(index, _)| index);
        start..end
    }

    /// The end of the word after `offset` (whitespace, then a word).
    pub(crate) fn next_word(&self, offset: usize) -> usize {
        let rest = &self.text[offset..];
        let mut chars = rest.char_indices().peekable();
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        while chars.next_if(|(_, c)| !c.is_whitespace()).is_some() {}
        offset + chars.peek().map_or(rest.len(), |(index, _)| *index)
    }

    /// Replaces `range` (bytes) with `new`, records it for undo, and
    /// returns the edit in code points (`None` when nothing changed).
    pub(crate) fn replace(
        &mut self,
        range: Range<usize>,
        new: &str,
        record: Record,
    ) -> Option<TextEdit> {
        let start = self.floor_boundary(range.start.min(range.end));
        let end = self.floor_boundary(range.end.max(range.start));
        if start == end && new.is_empty() {
            return None;
        }
        let edit = TextEdit {
            start: self.code_points(start),
            end: self.code_points(end),
            text: new.to_string(),
        };
        let removed = self.text[start..end].to_string();
        self.text.replace_range(start..end, new);
        self.move_to(start + new.len());
        self.push_undo(UndoEntry {
            start,
            removed,
            inserted: new.to_string(),
            kind: record,
        });
        self.redo.clear();
        Some(edit)
    }

    fn push_undo(&mut self, entry: UndoEntry) {
        if let Some(last) = self.undo.last_mut() {
            match (entry.kind, last.kind) {
                (Record::Typing, Record::Typing)
                    if entry.removed.is_empty()
                        && last.start + last.inserted.len() == entry.start
                        && !last.inserted.ends_with(char::is_whitespace) =>
                {
                    last.inserted.push_str(&entry.inserted);
                    return;
                }
                (Record::Deleting, Record::Deleting)
                    if entry.inserted.is_empty() && last.inserted.is_empty() =>
                {
                    if entry.start + entry.removed.len() == last.start {
                        last.start = entry.start;
                        last.removed.insert_str(0, &entry.removed);
                        return;
                    }
                    if entry.start == last.start {
                        last.removed.push_str(&entry.removed);
                        return;
                    }
                }
                _ => {}
            }
        }
        self.undo.push(entry);
        if self.undo.len() > UNDO_LIMIT {
            self.undo.remove(0);
        }
    }

    pub(crate) fn undo(&mut self) -> Option<TextEdit> {
        let entry = self.undo.pop()?;
        let range = entry.start..entry.start + entry.inserted.len();
        let edit = TextEdit {
            start: self.code_points(range.start),
            end: self.code_points(range.end),
            text: entry.removed.clone(),
        };
        self.text.replace_range(range, &entry.removed);
        self.move_to(entry.start + entry.removed.len());
        self.redo.push(entry);
        Some(edit)
    }

    pub(crate) fn redo(&mut self) -> Option<TextEdit> {
        let entry = self.redo.pop()?;
        let range = entry.start..entry.start + entry.removed.len();
        let edit = TextEdit {
            start: self.code_points(range.start),
            end: self.code_points(range.end),
            text: entry.inserted.clone(),
        };
        self.text.replace_range(range, &entry.inserted);
        self.move_to(entry.start + entry.inserted.len());
        self.undo.push(entry);
        Some(edit)
    }

    /// Adopts text that changed without the keyboard. The caret and
    /// selection move with the text around the change; undo steps that
    /// touched the changed span (and everything older) are dropped, since
    /// they could no longer be replayed faithfully. An `undoable` change
    /// (an accepted proposal, back to raw) becomes one undo step itself.
    /// Returns the byte mapping from old to new offsets, if anything
    /// changed.
    pub(crate) fn sync(
        &mut self,
        new: &str,
        undoable: bool,
    ) -> Option<impl Fn(usize) -> usize + use<>> {
        if new == self.text {
            return None;
        }
        let old = std::mem::replace(&mut self.text, new.to_string());
        let (prefix, old_end, new_end) = changed_span(&old, new);
        // Offsets before the change stay, offsets after it shift; at a
        // pure insertion point (prefix == old_end) they shift too, so a
        // caret or an edit sitting right after a growing tail stays after
        // it.
        let map = move |offset: usize| {
            if offset < prefix || (offset == prefix && prefix < old_end) {
                offset
            } else if offset >= old_end {
                offset - old_end + new_end
            } else {
                new_end
            }
        };
        let (start, end) = (map(self.selected.start), map(self.selected.end));
        self.selected = start.min(end)..start.max(end);
        // Mapping is monotone, so the existing anchor/head direction stays valid.
        let touches = |start: usize, len: usize| {
            if len == 0 {
                prefix < start && start < old_end
            } else {
                start < old_end && start + len > prefix
            }
        };
        let remap = |stack: &mut Vec<UndoEntry>, span: fn(&UndoEntry) -> usize| {
            if let Some(last_bad) = stack
                .iter()
                .rposition(|entry| touches(entry.start, span(entry)))
            {
                stack.drain(..=last_bad);
            }
            for entry in stack.iter_mut() {
                entry.start = map(entry.start);
            }
        };
        remap(&mut self.undo, |entry| entry.inserted.len());
        remap(&mut self.redo, |entry| entry.removed.len());
        if undoable {
            self.push_undo(UndoEntry {
                start: prefix,
                removed: old[prefix..old_end].to_string(),
                inserted: new[prefix..new_end].to_string(),
                kind: Record::Plain,
            });
            self.redo.clear();
        }
        Some(map)
    }
}

// ----------------------------------------------------------------------
// Layout: wrapped lines and the geometry the keys and the mouse need
// ----------------------------------------------------------------------

struct EditorLayout {
    shaped: Rc<ShapedText>,
    bounds: Bounds<Pixels>,
}

impl Deref for EditorLayout {
    type Target = ShapedText;
    fn deref(&self) -> &Self::Target {
        &self.shaped
    }
}

struct ShapedText {
    /// Paragraph byte offset, top, shaped text, and cached wrap starts.
    lines: Vec<(usize, Pixels, WrappedLine, Vec<usize>)>,
    rows: Vec<Range<usize>>,
    line_height: Pixels,
    height: Pixels,
}

/// Immutable text/style snapshot. Measurement and painting use exactly
/// the same shaping result, including the fallback on a shaping error.
struct ShapeInput {
    text: SharedString,
    empty: bool,
    kinds: Vec<(Range<usize>, RegionKind)>,
    marked: Option<Range<usize>>,
    style: gpui::TextStyle,
    font_size: Pixels,
    line_height: Pixels,
    layouts: RefCell<Vec<(Option<Pixels>, Rc<ShapedText>)>>,
    reported_error: Cell<bool>,
}

impl ShapeInput {
    fn shape(&self, width: Option<Pixels>, window: &mut Window) -> Rc<ShapedText> {
        if let Some((_, layout)) = self
            .layouts
            .borrow()
            .iter()
            .find(|(known, _)| *known == width)
        {
            return layout.clone();
        }
        let mut lines = Vec::new();
        let mut rows = Vec::new();
        let mut base = 0;
        let mut top = px(0.);
        for text in self.text.split('\n') {
            let end = base + text.len();
            let runs = if self.empty {
                vec![TextRun {
                    len: text.len(),
                    font: self.style.font(),
                    color: theme::DIM.into(),
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                }]
            } else {
                runs_for(base, end, &self.kinds, self.marked.as_ref(), &self.style)
            };
            let shaped = match window.text_system().shape_text(
                SharedString::from(text.to_string()),
                self.font_size,
                &runs,
                width,
                None,
            ) {
                Ok(lines) => lines.into_iter().next().unwrap_or_default(),
                Err(err) => {
                    if !self.reported_error.replace(true) {
                        eprintln!("Could not lay out staging text: {err}");
                    }
                    WrappedLine::default()
                }
            };
            let starts = EditorLayout::row_starts(&shaped);
            for (index, start) in starts.iter().enumerate() {
                rows.push(
                    base + start..base + starts.get(index + 1).copied().unwrap_or(shaped.len()),
                );
            }
            let count = starts.len();
            lines.push((base, top, shaped, starts));
            top += self.line_height * count as f32;
            base = end + 1;
        }
        let shaped = Rc::new(ShapedText {
            lines,
            rows,
            line_height: self.line_height,
            height: top,
        });
        // Layout may probe an unconstrained width before the final width.
        // Retain a small number of probes rather than grow during resizing.
        let mut layouts = self.layouts.borrow_mut();
        if layouts.len() == 3 {
            layouts.remove(0);
        }
        layouts.push((width, shaped.clone()));
        shaped
    }
}

impl EditorLayout {
    fn row_starts(line: &WrappedLine) -> Vec<usize> {
        let unwrapped = &line.unwrapped_layout;
        std::iter::once(0)
            .chain(
                line.wrap_boundaries().iter().map(|boundary| {
                    unwrapped.runs[boundary.run_ix].glyphs[boundary.glyph_ix].index
                }),
            )
            .collect()
    }

    /// The line holding byte `offset`.
    fn line_at(&self, offset: usize) -> Option<usize> {
        self.lines
            .iter()
            .rposition(|(base, _, _, _)| *base <= offset)
    }

    /// The caret position for byte `offset`, relative to the element.
    fn position_for(&self, offset: usize) -> Point<Pixels> {
        let Some(index) = self.line_at(offset) else {
            return point(px(0.), px(0.));
        };
        let (base, top, line, starts) = &self.lines[index];
        let local = (offset - base).min(line.len());
        let row = starts
            .iter()
            .rposition(|start| *start <= local)
            .unwrap_or(0);
        let unwrapped = &line.unwrapped_layout;
        let x = unwrapped.x_for_index(local) - unwrapped.x_for_index(starts[row]);
        point(x, *top + self.line_height * row as f32)
    }

    /// The byte offset closest to `position` (relative to the element).
    fn index_for(&self, position: Point<Pixels>) -> usize {
        let Some(index) = self
            .lines
            .iter()
            .rposition(|(_, top, _, _)| *top <= position.y)
            .or((!self.lines.is_empty()).then_some(0))
        else {
            return 0;
        };
        let (base, top, line, _) = &self.lines[index];
        let local = point(position.x.max(px(0.)), (position.y - *top).max(px(0.)));
        let found = match line.closest_index_for_position(local, self.line_height) {
            Ok(found) | Err(found) => found,
        };
        base + found.min(line.len())
    }

    /// Every visual row, computed once per shaped layout.
    fn rows(&self) -> &[Range<usize>] {
        &self.shaped.rows
    }

    /// The x of `offset` on the row starting at `row_start`, and that
    /// row's top.
    fn row_x(&self, row_start: usize, offset: usize) -> (Pixels, Pixels) {
        let Some(index) = self.line_at(row_start) else {
            return (px(0.), px(0.));
        };
        let (base, top, line, starts) = &self.lines[index];
        let local_start = row_start - base;
        let row = starts
            .iter()
            .position(|start| *start == local_start)
            .unwrap_or(0);
        let unwrapped = &line.unwrapped_layout;
        let local = (offset - base).min(line.len());
        (
            unwrapped.x_for_index(local) - unwrapped.x_for_index(local_start),
            *top + self.line_height * row as f32,
        )
    }

    fn row_of(&self, offset: usize) -> Option<Range<usize>> {
        let rows = self.rows();
        rows.iter()
            .find(|row| row.start <= offset && offset < row.end)
            .or_else(|| {
                rows.iter()
                    .rev()
                    .find(|row| row.start <= offset && offset <= row.end)
            })
            .cloned()
    }

    fn height(&self) -> Pixels {
        self.shaped.height
    }
}

// ----------------------------------------------------------------------
// The entity
// ----------------------------------------------------------------------

pub(crate) struct StagingEditor {
    focus_handle: FocusHandle,
    pub(crate) buffer: EditBuffer,
    /// Region kinds over the text, in bytes, for styling.
    kinds: Vec<(Range<usize>, RegionKind)>,
    marked_range: Option<Range<usize>>,
    placeholder: SharedString,
    layout: Option<EditorLayout>,
    shape_input: RefCell<Option<Rc<ShapeInput>>>,
    is_selecting: bool,
    pub(crate) scroll: ScrollHandle,
    autoscroll: bool,
    /// The x a run of Up/Down keeps aiming for.
    goal_x: Option<Pixels>,
}

impl EventEmitter<EditorEvent> for StagingEditor {}

impl StagingEditor {
    pub(crate) fn new(placeholder: &str, cx: &mut Context<Self>) -> StagingEditor {
        StagingEditor {
            focus_handle: cx.focus_handle(),
            buffer: EditBuffer::default(),
            kinds: Vec::new(),
            marked_range: None,
            placeholder: SharedString::from(placeholder.to_string()),
            layout: None,
            shape_input: RefCell::new(None),
            is_selecting: false,
            scroll: ScrollHandle::new(),
            autoscroll: true,
            goal_x: None,
        }
    }

    fn shape_input(&self, window: &Window) -> Rc<ShapeInput> {
        let empty = self.buffer.text.is_empty();
        let text: &str = if empty {
            &self.placeholder
        } else {
            &self.buffer.text
        };
        let style = window.text_style();
        let font_size = style.font_size.to_pixels(window.rem_size());
        let line_height = window.line_height();
        let mut cached = self.shape_input.borrow_mut();
        if let Some(input) = cached.as_ref() {
            if input.text.as_ref() == text
                && input.empty == empty
                && input.kinds == self.kinds
                && input.marked == self.marked_range
                && input.style == style
                && input.font_size == font_size
                && input.line_height == line_height
            {
                return input.clone();
            }
        }
        let input = Rc::new(ShapeInput {
            text: SharedString::from(text.to_string()),
            empty,
            kinds: self.kinds.clone(),
            marked: self.marked_range.clone(),
            style,
            font_size,
            line_height,
            layouts: RefCell::new(Vec::new()),
            reported_error: Cell::new(false),
        });
        *cached = Some(input.clone());
        input
    }

    /// Adopts the draft's text and region kinds (`kinds` in code points).
    /// Ranges come from the draft snapshot in sorted, non-overlapping order.
    pub(crate) fn set_content(
        &mut self,
        text: &str,
        kinds: &[(Range<usize>, RegionKind)],
        undoable: bool,
        cx: &mut Context<Self>,
    ) {
        let cursor_at_end = self.buffer.cursor() == self.buffer.text.len();
        if let Some(map) = self.buffer.sync(text, undoable) {
            self.marked_range = self
                .marked_range
                .take()
                .map(|range| map(range.start)..map(range.end))
                .filter(|range| !range.is_empty());
            // A caret parked at the end follows the live tail.
            if cursor_at_end && self.buffer.selected.is_empty() {
                self.buffer.move_to(self.buffer.text.len());
            }
            self.autoscroll |= cursor_at_end || undoable;
        }
        let mut byte = 0;
        let mut chars = 0;
        let mut iter = self.buffer.text.char_indices();
        let mut to_byte = |target: usize| {
            debug_assert!(target >= chars, "draft kind ranges must be ordered");
            while chars < target {
                match iter.next() {
                    Some((index, c)) => {
                        byte = index + c.len_utf8();
                        chars += 1;
                    }
                    None => break,
                }
            }
            byte
        };
        self.kinds = kinds
            .iter()
            .map(|(range, kind)| (to_byte(range.start)..to_byte(range.end), *kind))
            .collect();
        cx.notify();
    }

    fn edit(&mut self, range: Range<usize>, text: &str, record: Record, cx: &mut Context<Self>) {
        if let Some(edit) = self.buffer.replace(range, text, record) {
            self.marked_range = None;
            self.autoscroll = true;
            self.goal_x = None;
            cx.emit(EditorEvent::Edit(edit));
            cx.notify();
        }
    }

    /// Deletes the selection, or from the cursor to `target`.
    fn delete_to(&mut self, target: usize, cx: &mut Context<Self>) {
        let range = if self.buffer.selected.is_empty() {
            let cursor = self.buffer.cursor();
            cursor.min(target)..cursor.max(target)
        } else {
            self.buffer.selected.clone()
        };
        self.edit(range, "", Record::Deleting, cx);
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.buffer.move_to(offset);
        self.autoscroll = true;
        cx.notify();
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.buffer.select_to(offset);
        self.autoscroll = true;
        cx.notify();
    }

    /// The offset one visual row above (`-1`) or below (`1`) the cursor.
    fn vertical(&mut self, direction: f32) -> usize {
        let cursor = self.buffer.cursor();
        let Some(layout) = self.layout.as_ref() else {
            return cursor;
        };
        let at = layout.position_for(cursor);
        let x = *self.goal_x.get_or_insert(at.x);
        let y = at.y + layout.line_height * direction + layout.line_height / 2.;
        if y < px(0.) {
            return 0;
        }
        if y >= layout.height() {
            return self.buffer.text.len();
        }
        // Probe the center of the destination row. `height()` is the
        // exclusive lower edge; equality is already below the last row.
        layout.index_for(point(x, y))
    }

    /// The visual row the cursor is on, as a byte range.
    fn current_row(&self) -> Range<usize> {
        let cursor = self.buffer.cursor();
        self.layout
            .as_ref()
            .and_then(|layout| layout.row_of(cursor))
            .unwrap_or(0..self.buffer.text.len())
    }

    /// End of the cursor's row: before the space a wrap left at its end,
    /// so the caret stays on the row it was asked for.
    fn row_end(&self) -> usize {
        let row = self.current_row();
        let text = &self.buffer.text[row.clone()];
        if row.end < self.buffer.text.len() && !self.buffer.text[row.end..].starts_with('\n') {
            row.start + text.trim_end().len()
        } else {
            row.end
        }
    }

    fn index_for_mouse(&self, position: Point<Pixels>) -> usize {
        match self.layout.as_ref() {
            Some(layout) => layout.index_for(position - layout.bounds.origin),
            None => self.buffer.text.len(),
        }
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle);
        let offset = self.index_for_mouse(event.position);
        self.goal_x = None;
        if event.click_count >= 2 {
            let word = self.buffer.word_range(offset);
            self.buffer.move_to(word.start);
            self.buffer.select_to(word.end);
            cx.notify();
            return;
        }
        self.is_selecting = true;
        if event.modifiers.shift {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx);
        }
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            let offset = self.index_for_mouse(event.position);
            self.select_to(offset, cx);
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn utf16_to_byte(&self, offset: usize) -> usize {
        let mut utf16 = 0;
        for (index, c) in self.buffer.text.char_indices() {
            if utf16 + c.len_utf16() > offset {
                return index;
            }
            utf16 += c.len_utf16();
        }
        self.buffer.text.len()
    }

    fn byte_to_utf16(&self, offset: usize) -> usize {
        self.buffer.text[..offset.min(self.buffer.text.len())]
            .chars()
            .map(char::len_utf16)
            .sum()
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.utf16_to_byte(range.start)..self.utf16_to_byte(range.end)
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.byte_to_utf16(range.start)..self.byte_to_utf16(range.end)
    }
}

impl EntityInputHandler for StagingEditor {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.buffer.text[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.buffer.selected),
            reversed: self.buffer.reversed,
        })
    }

    fn marked_text_range(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.buffer.selected.clone());
        self.edit(range, new_text, Record::Typing, cx);
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.buffer.selected.clone());
        self.replace_text_in_range(range_utf16, new_text, window, cx);
        if !new_text.is_empty() {
            self.marked_range = Some(range.start..range.start + new_text.len());
        }
        if let Some(selected) = new_selected_range_utf16 {
            let base = self.byte_to_utf16(range.start);
            let selected = self.range_from_utf16(&(base + selected.start..base + selected.end));
            self.buffer.selected = selected;
            cx.notify();
        }
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        _bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let layout = self.layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        let start = layout.position_for(range.start);
        let end = layout.position_for(range.end);
        let origin = layout.bounds.origin;
        Some(Bounds::from_corners(
            origin + start,
            origin + point(end.x.max(start.x), end.y + layout.line_height),
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let offset = self.index_for_mouse(point);
        Some(self.byte_to_utf16(offset))
    }
}

impl Focusable for StagingEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for StagingEditor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("staging-editor")
            .key_context(CONTEXT)
            .track_focus(&self.focus_handle(cx))
            .track_scroll(&self.scroll)
            .overflow_y_scroll()
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(|this, _: &Backspace, _, cx| {
                let target = this.buffer.previous_boundary(this.buffer.cursor());
                this.delete_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &Delete, _, cx| {
                let target = this.buffer.next_boundary(this.buffer.cursor());
                this.delete_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &DeleteWordLeft, _, cx| {
                let target = this.buffer.previous_word(this.buffer.cursor());
                this.delete_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &DeleteWordRight, _, cx| {
                let target = this.buffer.next_word(this.buffer.cursor());
                this.delete_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &DeleteLine, _, cx| {
                let row = this.current_row();
                // A visual row ending at a logical line boundary takes its newline with it.
                let end = if this.buffer.text[row.end..].starts_with('\n') {
                    row.end + 1
                } else {
                    row.end
                };
                this.edit(row.start..end, "", Record::Plain, cx);
            }))
            .on_action(cx.listener(|this, _: &Left, _, cx| {
                this.goal_x = None;
                let target = if this.buffer.selected.is_empty() {
                    this.buffer.previous_boundary(this.buffer.cursor())
                } else {
                    this.buffer.selected.start
                };
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &Right, _, cx| {
                this.goal_x = None;
                let target = if this.buffer.selected.is_empty() {
                    this.buffer.next_boundary(this.buffer.cursor())
                } else {
                    this.buffer.selected.end
                };
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &Up, _, cx| {
                let target = this.vertical(-1.);
                this.buffer.move_to(target);
                this.autoscroll = true;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Down, _, cx| {
                let target = this.vertical(1.);
                this.buffer.move_to(target);
                this.autoscroll = true;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &SelectUp, _, cx| {
                let target = this.vertical(-1.);
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectDown, _, cx| {
                let target = this.vertical(1.);
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &WordLeft, _, cx| {
                this.goal_x = None;
                let target = this.buffer.previous_word(this.buffer.cursor());
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &WordRight, _, cx| {
                this.goal_x = None;
                let target = this.buffer.next_word(this.buffer.cursor());
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectLeft, _, cx| {
                this.goal_x = None;
                let target = this.buffer.previous_boundary(this.buffer.cursor());
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectRight, _, cx| {
                this.goal_x = None;
                let target = this.buffer.next_boundary(this.buffer.cursor());
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectWordLeft, _, cx| {
                this.goal_x = None;
                let target = this.buffer.previous_word(this.buffer.cursor());
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectWordRight, _, cx| {
                this.goal_x = None;
                let target = this.buffer.next_word(this.buffer.cursor());
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &Home, _, cx| {
                this.goal_x = None;
                let target = this.current_row().start;
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &End, _, cx| {
                this.goal_x = None;
                let target = this.row_end();
                this.move_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectHome, _, cx| {
                let target = this.current_row().start;
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectEnd, _, cx| {
                let target = this.row_end();
                this.select_to(target, cx);
            }))
            .on_action(cx.listener(|this, _: &DocStart, _, cx| this.move_to(0, cx)))
            .on_action(cx.listener(|this, _: &DocEnd, _, cx| {
                let end = this.buffer.text.len();
                this.move_to(end, cx);
            }))
            .on_action(cx.listener(|this, _: &SelectAll, _, cx| {
                this.buffer.move_to(0);
                let end = this.buffer.text.len();
                this.buffer.select_to(end);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Newline, _, cx| {
                let range = this.buffer.selected.clone();
                this.edit(range, "\n", Record::Plain, cx);
            }))
            .on_action(cx.listener(|this, _: &Undo, _, cx| {
                if let Some(edit) = this.buffer.undo() {
                    this.marked_range = None;
                    this.autoscroll = true;
                    cx.emit(EditorEvent::Edit(edit));
                    cx.notify();
                }
            }))
            .on_action(cx.listener(|this, _: &Redo, _, cx| {
                if let Some(edit) = this.buffer.redo() {
                    this.marked_range = None;
                    this.autoscroll = true;
                    cx.emit(EditorEvent::Edit(edit));
                    cx.notify();
                }
            }))
            .on_action(cx.listener(|this, _: &Copy, _, cx| {
                if !this.buffer.selected.is_empty() {
                    let text = this.buffer.text[this.buffer.selected.clone()].to_string();
                    cx.write_to_clipboard(ClipboardItem::new_string(text));
                }
            }))
            .on_action(cx.listener(|this, _: &Cut, _, cx| {
                if !this.buffer.selected.is_empty() {
                    let range = this.buffer.selected.clone();
                    cx.write_to_clipboard(ClipboardItem::new_string(
                        this.buffer.text[range.clone()].to_string(),
                    ));
                    this.edit(range, "", Record::Plain, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &Paste, _, cx| {
                if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
                    let range = this.buffer.selected.clone();
                    this.edit(range, &text.replace("\r\n", "\n"), Record::Plain, cx);
                }
            }))
            .on_action(cx.listener(|_, _: &Submit, _, cx| cx.emit(EditorEvent::Submit)))
            .on_action(cx.listener(|_, _: &Leave, _, cx| cx.emit(EditorEvent::Leave)))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .size_full()
            .child(EditorElement {
                editor: cx.entity(),
            })
    }
}

// ----------------------------------------------------------------------
// The element
// ----------------------------------------------------------------------

struct EditorElement {
    editor: Entity<StagingEditor>,
}

struct EditorPrepaint {
    layout: Option<EditorLayout>,
    selections: Vec<PaintQuad>,
    cursor: Option<PaintQuad>,
}

impl IntoElement for EditorElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The text runs for `[start, end)` of the text: one per region-kind
/// change, the live tail dim and italic, IME composition underlined.
fn runs_for(
    start: usize,
    end: usize,
    kinds: &[(Range<usize>, RegionKind)],
    marked: Option<&Range<usize>>,
    base: &gpui::TextStyle,
) -> Vec<TextRun> {
    let mut cuts: Vec<usize> = vec![start, end];
    for (range, _) in kinds {
        cuts.extend([range.start, range.end]);
    }
    if let Some(marked) = marked {
        cuts.extend([marked.start, marked.end]);
    }
    cuts.retain(|cut| (start..=end).contains(cut));
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2)
        .map(|pair| {
            let (from, to) = (pair[0], pair[1]);
            let kind = kinds
                .iter()
                .find(|(range, _)| range.start <= from && from < range.end)
                .map(|(_, kind)| *kind);
            let mut font = base.font();
            let color: Hsla = match kind {
                Some(RegionKind::Partial) => {
                    font.style = FontStyle::Italic;
                    theme::MUTED.into()
                }
                Some(RegionKind::Command) => theme::EYEBROW.into(),
                _ => base.color,
            };
            let underline = marked
                .filter(|marked| marked.start <= from && from < marked.end)
                .map(|_| UnderlineStyle {
                    thickness: px(1.),
                    color: Some(color),
                    wavy: false,
                });
            TextRun {
                len: to - from,
                font,
                color,
                background_color: None,
                underline,
                strikethrough: None,
            }
        })
        .collect()
}

impl Element for EditorElement {
    type RequestLayoutState = ();
    type PrepaintState = EditorPrepaint;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let input = self.editor.read(cx).shape_input(window);
        let mut layout_style = Style::default();
        layout_style.size.width = relative(1.).into();
        let id =
            window.request_measured_layout(layout_style, move |known, available, window, _cx| {
                let width = known.width.or(match available.width {
                    gpui::AvailableSpace::Definite(width) => Some(width),
                    _ => None,
                });
                let shaped = input.shape(width, window);
                size(width.unwrap_or(px(0.)), shaped.height)
            });
        (id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let editor = self.editor.read(cx);
        let empty = editor.buffer.text.is_empty();
        let input = editor.shape_input(window);
        let layout = EditorLayout {
            shaped: input.shape(Some(bounds.size.width), window),
            bounds,
        };
        let line_height = layout.line_height;

        let selected = editor.buffer.selected.clone();
        let mut selections = Vec::new();
        let mut cursor = None;
        if empty || selected.is_empty() {
            let at = if empty {
                point(px(0.), px(0.))
            } else {
                layout.position_for(editor.buffer.cursor())
            };
            cursor = Some(fill(
                Bounds::new(bounds.origin + at, size(px(1.5), line_height)),
                theme::LIME,
            ));
        } else {
            for row in layout.rows() {
                let start = selected.start.max(row.start);
                let end = selected.end.min(row.end);
                if start >= end {
                    continue;
                }
                let (from, top) = layout.row_x(row.start, start);
                let (to, _) = layout.row_x(row.start, end);
                selections.push(fill(
                    Bounds::from_corners(
                        bounds.origin + point(from, top),
                        bounds.origin + point(to.max(from + px(4.)), top + line_height),
                    ),
                    theme::STAGING_SELECTION,
                ));
            }
        }
        EditorPrepaint {
            layout: Some(layout),
            selections,
            cursor,
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.editor.read(cx).focus_handle.clone();
        window.handle_input(
            &focus_handle,
            ElementInputHandler::new(bounds, self.editor.clone()),
            cx,
        );
        for selection in prepaint.selections.drain(..) {
            window.paint_quad(selection);
        }
        let Some(layout) = prepaint.layout.take() else {
            return;
        };
        for (_, top, line, _) in &layout.lines {
            line.paint(
                bounds.origin + point(px(0.), *top),
                layout.line_height,
                gpui::TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
        }
        let focused = focus_handle.is_focused(window);
        let cursor = prepaint.cursor.take();
        if focused {
            if let Some(cursor) = cursor.clone() {
                window.paint_quad(cursor);
            }
        }
        self.editor.update(cx, |editor, cx| {
            // Keep the caret in view after an edit or a move: the scroll
            // viewport is the editor's own scroll container.
            if editor.autoscroll {
                editor.autoscroll = false;
                if let Some(cursor) = cursor {
                    let viewport = editor.scroll.bounds();
                    let offset = editor.scroll.offset();
                    let caret = cursor.bounds;
                    let mut shift = px(0.);
                    if caret.bottom() > viewport.bottom() {
                        shift = viewport.bottom() - caret.bottom() - px(8.);
                    } else if caret.top() < viewport.top() {
                        shift = viewport.top() - caret.top();
                    }
                    if shift != px(0.) && viewport.size != Size::default() {
                        editor
                            .scroll
                            .set_offset(point(offset.x, (offset.y + shift).min(px(0.))));
                        cx.notify();
                    }
                }
            }
            editor.layout = Some(layout);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(text: &str) -> EditBuffer {
        let mut buffer = EditBuffer {
            text: text.to_string(),
            ..EditBuffer::default()
        };
        buffer.move_to(text.len());
        buffer
    }

    fn typed(buffer: &mut EditBuffer, text: &str) {
        for c in text.chars() {
            let at = buffer.cursor();
            buffer.replace(at..at, &c.to_string(), Record::Typing);
        }
    }

    #[gpui::test]
    fn layout_reuses_shaping_and_caches_visual_rows(cx: &mut gpui::TestAppContext) {
        let window = cx.add_window(|_, cx| StagingEditor::new("", cx));
        window
            .update(cx, |editor, window, cx| {
                let text = "one two three four five six seven eight nine ten\nlast line";
                editor.set_content(text, &[(0..text.len(), RegionKind::Partial)], false, cx);
                let input = editor.shape_input(window);
                let measured = input.shape(Some(px(90.)), window);
                let painted = editor.shape_input(window).shape(Some(px(90.)), window);
                assert!(
                    Rc::ptr_eq(&measured, &painted),
                    "measurement and paint must share shaping"
                );
                assert!(measured.rows.len() > 2, "the first paragraph must wrap");
                let layout = EditorLayout {
                    shaped: painted,
                    bounds: Bounds::new(point(px(0.), px(0.)), size(px(90.), measured.height)),
                };
                assert_eq!(layout.rows().last().unwrap().end, text.len());
                editor.layout = Some(layout);
                editor.buffer.move_to(0);
                assert_eq!(editor.vertical(-1.), 0);
                editor.buffer.move_to(text.len());
                assert_eq!(editor.vertical(1.), text.len());
                editor.buffer.move_to(0);
                let row = editor.current_row();
                assert!(
                    row.end < text.find('\n').unwrap(),
                    "delete-line targets one visual row"
                );
                assert!(
                    Rc::ptr_eq(&input, &editor.shape_input(window)),
                    "caret movement must reuse the snapshot"
                );
                assert!(!Rc::ptr_eq(&measured, &input.shape(Some(px(180.)), window)));
                editor.set_content("different", &[], false, cx);
                assert!(!Rc::ptr_eq(&input, &editor.shape_input(window)));
            })
            .unwrap();
    }

    #[test]
    fn external_undo_entries_respect_the_limit() {
        let mut buffer = buffer("");
        for _ in 0..UNDO_LIMIT + 20 {
            let next = format!("{}x", buffer.text);
            buffer.sync(&next, true);
        }
        assert_eq!(buffer.undo.len(), UNDO_LIMIT);
        for _ in 0..UNDO_LIMIT {
            buffer.undo().unwrap();
        }
        assert_eq!(buffer.text, "x".repeat(20));
    }

    #[test]
    fn external_text_preserves_a_reversed_selection() {
        let mut buffer = buffer("abc def");
        buffer.move_to(7);
        buffer.select_to(4);
        buffer.sync("123 abc def", false);
        assert!(buffer.reversed);
        assert_eq!(buffer.cursor(), 8);
        buffer.select_to(9);
        assert_eq!(buffer.selected, 9..11);
    }

    #[gpui::test]
    fn ime_offsets_inside_surrogates_clamp_to_the_character(cx: &mut gpui::TestAppContext) {
        let editor = cx.new(|cx| StagingEditor::new("", cx));
        editor.update(cx, |editor, _| {
            editor.buffer = buffer("a😀b");
            assert_eq!(editor.utf16_to_byte(2), 1);
            assert_eq!(editor.utf16_to_byte(3), 5);
            assert_eq!(editor.utf16_to_byte(99), 6);
        });
    }

    #[test]
    fn edits_are_reported_in_code_points() {
        let mut buffer = buffer("héllo wörld");
        let start = "héllo ".len();
        let edit = buffer
            .replace(start..buffer.text.len(), "there", Record::Plain)
            .unwrap();
        assert_eq!(
            edit,
            TextEdit {
                start: 6,
                end: 11,
                text: "there".into()
            }
        );
        assert_eq!(buffer.text, "héllo there");
    }

    #[test]
    fn typing_undoes_a_word_at_a_time() {
        let mut buffer = buffer("");
        typed(&mut buffer, "one two");
        assert_eq!(buffer.undo().unwrap().text, "");
        assert_eq!(buffer.text, "one ");
        buffer.undo();
        assert_eq!(buffer.text, "");
        assert_eq!(buffer.redo().unwrap().text, "one ");
        assert_eq!(buffer.text, "one ");
    }

    #[test]
    fn repeated_backspace_undoes_in_one_step() {
        let mut buffer = buffer("hello");
        for _ in 0..3 {
            let cursor = buffer.cursor();
            let target = buffer.previous_boundary(cursor);
            buffer.replace(target..cursor, "", Record::Deleting);
        }
        assert_eq!(buffer.text, "he");
        let undo = buffer.undo().unwrap();
        assert_eq!((undo.start, undo.end, undo.text.as_str()), (2, 2, "llo"));
        assert_eq!(buffer.text, "hello");
    }

    #[test]
    fn word_motion_skips_spaces_then_a_word() {
        let buffer = buffer("say it  as you mean");
        assert_eq!(
            buffer.previous_word(buffer.text.len()),
            "say it  as you ".len()
        );
        assert_eq!(buffer.previous_word("say it  ".len()), "say ".len());
        assert_eq!(buffer.next_word(3), "say it".len());
        assert_eq!(buffer.next_word("say it".len()), "say it  as".len());
        assert_eq!(buffer.previous_word(0), 0);
        assert_eq!(buffer.next_word(buffer.text.len()), buffer.text.len());
    }

    #[test]
    fn a_live_change_before_the_caret_moves_the_caret_and_the_history() {
        // The user typed " (note)" after a live partial; the partial then
        // grows in front of it.
        let mut buffer = buffer("hello wor");
        typed(&mut buffer, " (note)");
        buffer.move_to("hello wor (no".len());
        buffer.sync("hello world, and (note)", false);
        assert_eq!(buffer.cursor(), "hello world, and (no".len());
        let undo = buffer.undo().unwrap();
        assert_eq!(buffer.text, "hello world, and ");
        assert_eq!(undo.start, "hello world, and ".chars().count());
    }

    #[test]
    fn a_live_change_over_an_edit_drops_that_history() {
        let mut buffer = buffer("the quick fox");
        buffer.replace(4..9, "slow", Record::Plain);
        buffer.sync("the brown fox", false);
        assert!(buffer.undo().is_none());
    }

    #[test]
    fn a_caret_at_the_end_follows_the_growing_tail() {
        let mut buffer = buffer("one two");
        buffer.sync("one two three", false);
        assert_eq!(buffer.cursor(), buffer.text.len());
    }

    #[test]
    fn an_accepted_proposal_is_one_undo_step() {
        let mut buffer = buffer("um so the thing is");
        buffer.sync("The thing is.", true);
        let undo = buffer.undo().unwrap();
        assert_eq!(buffer.text, "um so the thing is");
        assert_eq!(undo.start, 0);
        assert_eq!(undo.text, "um so the thing is");
        assert_eq!(buffer.redo().unwrap().text, "The thing is.");
        assert_eq!(buffer.text, "The thing is.");
    }

    #[test]
    fn a_double_click_word_is_the_run_of_non_spaces() {
        let buffer = buffer("say it, now");
        assert_eq!(buffer.word_range(5), 4..7);
        assert_eq!(buffer.word_range(4), 4..7);
        assert_eq!(buffer.word_range(0), 0..3);
    }

    #[test]
    fn changed_spans_stay_on_char_boundaries() {
        assert_eq!(changed_span("aé", "aè"), (1, 3, 3));
        assert_eq!(changed_span("abc", "abc d"), (3, 3, 5));
        assert_eq!(changed_span("aaa", "aa"), (2, 3, 2));
    }
}
