//! The text before the caret of the focused field, over AT-SPI (Linux,
//! #341): what the insertion-boundary rules need, without an input method.
//!
//! # Which field
//!
//! [`AtspiReader::locate`] runs when a target is captured. Among the
//! applications on the accessibility bus that may own the target (its
//! pid, where the backend knows it, X11; any, without one, Wayland) it
//! needs exactly one active window, and in it exactly one object with
//! `STATE_FOCUSED`: through `Collection.GetMatches` where the toolkit
//! implements it, else a walk of the showing tree with a node and time
//! budget. Two candidates, none, or a call that fails on the way find
//! nothing: a guess could name another app's field. Excluded pids (always
//! Starling) never count.
//!
//! [`AtspiReader::read`] reads only that object, and only while it is
//! still focused, editable and owned by the same process (and, where the
//! target has a pid, by the target's). A field focused later is never
//! read, even in the same window.
//!
//! # What is read
//!
//! Only the text before the insertion point (the caret, or the start of
//! a selection the text will replace), at most [`BEFORE_CHARS`]
//! characters: no rule reads the text after it. A password field
//! (`ROLE_PASSWORD_TEXT`) answers [`Surrounding::Protected`] before any
//! text call is made. AT-SPI has no "secret" state besides that role
//! (`STATE_SENSITIVE` means "enabled", which every usable field is).
//!
//! # Reach
//!
//! The bus is `AT_SPI_BUS_ADDRESS`, else the address `org.a11y.Bus` on
//! the session bus hands out. GTK 3/4 apps answer out of the box; Qt,
//! Chromium and Firefox expose their trees only while accessibility is
//! switched on (a screen reader, or `org.a11y.Status.IsEnabled`, which
//! this reader never sets). Without a bus, or with a toolkit that does
//! not expose the field's `Text`, the answer is
//! [`Surrounding::Unsupported`] and the text goes in as dictated.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use zbus::blocking::Connection;
use zbus::zvariant::{OwnedObjectPath, OwnedValue};

use crate::{FieldAnchor, FieldReader, Surrounding, SurroundingText, TargetSnapshot};

/// The most characters before the insertion point ever read.
pub const BEFORE_CHARS: usize = 128;

/// Every AT-SPI call's limit: a hung application costs one call, not a
/// delivery.
const CALL_TIMEOUT: Duration = Duration::from_millis(300);
/// The tree walk's budgets, when `Collection` is not implemented.
const WALK_NODES: usize = 1500;
const WALK_TIME: Duration = Duration::from_millis(800);

const REGISTRY: &str = "org.a11y.atspi.Registry";
const ROOT_PATH: &str = "/org/a11y/atspi/accessible/root";
const ACCESSIBLE: &str = "org.a11y.atspi.Accessible";
const COLLECTION: &str = "org.a11y.atspi.Collection";
const TEXT: &str = "org.a11y.atspi.Text";

/// `AtspiRole` values.
pub const ROLE_PASSWORD_TEXT: u32 = 40;

/// `AtspiStateType` values.
pub const STATE_ACTIVE: u32 = 1;
pub const STATE_EDITABLE: u32 = 7;
pub const STATE_FOCUSED: u32 = 12;
pub const STATE_SHOWING: u32 = 25;
pub const STATE_MANAGES_DESCENDANTS: u32 = 31;
pub const STATE_READ_ONLY: u32 = 43;

/// `AtspiCollectionMatchType::ALL` and `AtspiCollectionSortOrder::CANONICAL`.
const MATCH_ALL: i32 = 1;
const SORT_CANONICAL: u32 = 1;

/// An AT-SPI state set (`au`, one bit per `AtspiStateType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct States(u64);

impl States {
    pub fn of(states: &[u32]) -> States {
        States(states.iter().fold(0, |bits, state| bits | 1 << state))
    }

    fn from_words(words: &[u32]) -> States {
        let low = u64::from(words.first().copied().unwrap_or(0));
        let high = u64::from(words.get(1).copied().unwrap_or(0));
        States(low | high << 32)
    }

    /// The wire form (`au`/`ai`, two words).
    pub fn words(self) -> Vec<u32> {
        vec![self.0 as u32, (self.0 >> 32) as u32]
    }

    pub fn has(self, state: u32) -> bool {
        self.0 & 1 << state != 0
    }
}

/// An accessible object: the bus name of its application and its path.
type Object = (String, OwnedObjectPath);

/// Opens the accessibility bus.
type Connect = Box<dyn Fn() -> zbus::Result<Connection> + Send + Sync>;

/// [`FieldReader`] over the AT-SPI accessibility bus. One connection,
/// opened on first use and again after any call fails.
pub struct AtspiReader {
    connect: Connect,
    connection: Mutex<Option<Connection>>,
    excluded_pids: Vec<u32>,
}

impl AtspiReader {
    /// The session's accessibility bus. `excluded_pids` are never located
    /// (the caller passes Starling's own).
    pub fn new(excluded_pids: Vec<u32>) -> AtspiReader {
        AtspiReader::with_connect(
            Box::new(|| {
                let address = match std::env::var("AT_SPI_BUS_ADDRESS") {
                    Ok(address) if !address.is_empty() => address,
                    _ => {
                        let session = zbus::blocking::connection::Builder::session()?
                            .method_timeout(CALL_TIMEOUT)
                            .build()?;
                        let reply = session.call_method(
                            Some("org.a11y.Bus"),
                            "/org/a11y/bus",
                            Some("org.a11y.Bus"),
                            "GetAddress",
                            &(),
                        )?;
                        reply.body().deserialize::<String>()?
                    }
                };
                open(&address)
            }),
            excluded_pids,
        )
    }

    /// The bus at `address` (tests: a private bus with scripted apps).
    pub fn with_address(address: String, excluded_pids: Vec<u32>) -> AtspiReader {
        AtspiReader::with_connect(Box::new(move || open(&address)), excluded_pids)
    }

    fn with_connect(connect: Connect, excluded_pids: Vec<u32>) -> AtspiReader {
        AtspiReader {
            connect,
            connection: Mutex::new(None),
            excluded_pids,
        }
    }

    /// Runs `body` on the connection; a failure drops it, so the next
    /// call reconnects (the bus or the registry may have restarted). An
    /// error an application answered with says nothing about the bus,
    /// and keeps it.
    fn with_bus<T>(&self, body: impl FnOnce(&Bus<'_>) -> zbus::Result<T>) -> Option<T> {
        let connection = {
            let mut slot = self
                .connection
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if slot.is_none() {
                match (self.connect)() {
                    Ok(connection) => *slot = Some(connection),
                    Err(error) => {
                        log::debug!("AT-SPI bus unreachable: {error}");
                        return None;
                    }
                }
            }
            slot.clone()?
        };
        match body(&Bus(&connection)) {
            Ok(value) => Some(value),
            Err(error @ zbus::Error::MethodError(..)) => {
                log::debug!("AT-SPI call failed: {error}");
                None
            }
            Err(error) => {
                log::debug!("AT-SPI call failed: {error}");
                *self
                    .connection
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = None;
                None
            }
        }
    }

    /// The focused object of the target's application; see the module
    /// docs. `Ok(None)` when there is none or more than one candidate.
    fn locate_on(
        &self,
        bus: &Bus<'_>,
        target_pid: Option<u32>,
    ) -> zbus::Result<Option<FieldAnchor>> {
        // Every eligible active window first, so one that cannot be
        // searched never hands the answer to another. Any failed call
        // ends the search without an answer (an error here).
        let mut active: Vec<(String, OwnedObjectPath, u32)> = Vec::new();
        for (app, app_path) in bus.children(REGISTRY, ROOT_PATH)? {
            let pid = bus.pid(&app)?;
            if self.excluded_pids.contains(&pid) || target_pid.is_some_and(|target| target != pid) {
                continue;
            }
            for (window_app, window) in bus.children(&app, app_path.as_str())? {
                if window_app == app && bus.states(&app, window.as_str())?.has(STATE_ACTIVE) {
                    active.push((app.clone(), window, pid));
                }
            }
        }
        let [(app, window, pid)] = active.as_slice() else {
            if !active.is_empty() {
                log::debug!("AT-SPI: more than one active window; reading none");
            }
            return Ok(None);
        };
        Ok(bus
            .focused_in(app, window.as_str())?
            .map(|path| FieldAnchor {
                bus_name: app.clone(),
                path: path.to_string(),
                pid: *pid,
            }))
    }

    fn read_on(
        &self,
        bus: &Bus<'_>,
        target: &TargetSnapshot,
        field: &FieldAnchor,
    ) -> zbus::Result<Surrounding> {
        if target.pid.is_some_and(|pid| pid != field.pid) || bus.pid(&field.bus_name)? != field.pid
        {
            return Ok(Surrounding::Unsupported);
        }
        let (name, path) = (field.bus_name.as_str(), field.path.as_str());
        // The role first: nothing of a password field is read.
        if bus.role(name, path)? == ROLE_PASSWORD_TEXT {
            return Ok(Surrounding::Protected);
        }
        let states = bus.states(name, path)?;
        if !states.has(STATE_FOCUSED) || !states.has(STATE_EDITABLE) || states.has(STATE_READ_ONLY)
        {
            return Ok(Surrounding::Unsupported);
        }
        let caret: i32 = bus.property(name, path, TEXT, "CaretOffset")?;
        if caret < 0 {
            return Ok(Surrounding::Unsupported);
        }
        // Typing replaces a selection, so the text goes in at its start.
        let mut insertion = caret;
        if bus.call::<_, i32>(name, path, TEXT, "GetNSelections", &())? > 0 {
            let (start, end): (i32, i32) = bus.call(name, path, TEXT, "GetSelection", &(0i32,))?;
            if start != end {
                insertion = start.min(end).max(0);
            }
        }
        let start = insertion.saturating_sub(BEFORE_CHARS as i32).max(0);
        let before: String = bus.call(name, path, TEXT, "GetText", &(start, insertion))?;
        // The calls are separate: a caret that moved, or focus that left,
        // meanwhile makes the text stale.
        let caret_after: i32 = bus.property(name, path, TEXT, "CaretOffset")?;
        if caret_after != caret || !bus.states(name, path)?.has(STATE_FOCUSED) {
            return Ok(Surrounding::Unsupported);
        }
        Ok(Surrounding::Text(SurroundingText {
            before: last_chars(&before, BEFORE_CHARS).to_string(),
            after: String::new(),
            selection: None,
        }))
    }
}

impl FieldReader for AtspiReader {
    fn locate(&self, target: &TargetSnapshot) -> Option<FieldAnchor> {
        self.with_bus(|bus| self.locate_on(bus, target.pid))
            .flatten()
    }

    fn read(&self, target: &TargetSnapshot, field: &FieldAnchor) -> Surrounding {
        self.with_bus(|bus| self.read_on(bus, target, field))
            .unwrap_or(Surrounding::Unsupported)
    }
}

fn open(address: &str) -> zbus::Result<Connection> {
    zbus::blocking::connection::Builder::address(address)?
        .method_timeout(CALL_TIMEOUT)
        .build()
}

/// The last `count` characters of `text` (a toolkit counting offsets in
/// UTF-16 units can hand back a few more).
fn last_chars(text: &str, count: usize) -> &str {
    match text.char_indices().rev().nth(count.saturating_sub(1)) {
        Some((index, _)) if count > 0 => &text[index..],
        _ if count == 0 => "",
        _ => text,
    }
}

/// The AT-SPI calls this reader makes.
struct Bus<'a>(&'a Connection);

/// A Collection match rule: `(states, match, attributes, match, roles,
/// match, interfaces, match, invert)`.
type MatchRule = (
    Vec<i32>,
    i32,
    HashMap<String, String>,
    i32,
    Vec<i32>,
    i32,
    Vec<String>,
    i32,
    bool,
);

impl Bus<'_> {
    fn call<B, R>(
        &self,
        name: &str,
        path: &str,
        interface: &str,
        method: &str,
        body: &B,
    ) -> zbus::Result<R>
    where
        B: serde::Serialize + zbus::zvariant::DynamicType,
        R: for<'d> zbus::zvariant::DynamicDeserialize<'d>,
    {
        let reply = self
            .0
            .call_method(Some(name), path, Some(interface), method, body)?;
        reply.body().deserialize::<R>()
    }

    fn property<T>(
        &self,
        name: &str,
        path: &str,
        interface: &str,
        property: &str,
    ) -> zbus::Result<T>
    where
        T: TryFrom<OwnedValue>,
        T::Error: Into<zbus::zvariant::Error>,
    {
        let value: OwnedValue = self.call(
            name,
            path,
            "org.freedesktop.DBus.Properties",
            "Get",
            &(interface, property),
        )?;
        T::try_from(value).map_err(|error| zbus::Error::Variant(error.into()))
    }

    fn pid(&self, name: &str) -> zbus::Result<u32> {
        self.call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "GetConnectionUnixProcessID",
            &(name,),
        )
    }

    fn children(&self, name: &str, path: &str) -> zbus::Result<Vec<Object>> {
        self.call(name, path, ACCESSIBLE, "GetChildren", &())
    }

    fn states(&self, name: &str, path: &str) -> zbus::Result<States> {
        let words: Vec<u32> = self.call(name, path, ACCESSIBLE, "GetState", &())?;
        Ok(States::from_words(&words))
    }

    fn role(&self, name: &str, path: &str) -> zbus::Result<u32> {
        self.call(name, path, ACCESSIBLE, "GetRole", &())
    }

    /// The focused object under `window` of application `name`.
    fn focused_in(&self, name: &str, window: &str) -> zbus::Result<Option<OwnedObjectPath>> {
        let rule: MatchRule = (
            States::of(&[STATE_FOCUSED])
                .words()
                .into_iter()
                .map(|word| word as i32)
                .collect(),
            MATCH_ALL,
            HashMap::new(),
            MATCH_ALL,
            Vec::new(),
            MATCH_ALL,
            Vec::new(),
            MATCH_ALL,
            false,
        );
        match self.call::<_, Vec<Object>>(
            name,
            window,
            COLLECTION,
            "GetMatches",
            &(rule, SORT_CANONICAL, 0i32, true),
        ) {
            Ok(matches) => {
                // Objects of the same application only; a toolkit that
                // answers with a stale entry is checked once more.
                let mut focused = matches
                    .into_iter()
                    .filter(|(owner, _)| owner == name)
                    .map(|(_, path)| path);
                let first = focused.next();
                if focused.next().is_some() {
                    return Ok(None);
                }
                if let Some(path) = &first {
                    if !self.states(name, path.as_str())?.has(STATE_FOCUSED) {
                        return Ok(None);
                    }
                }
                Ok(first)
            }
            // No Collection (GTK 4, some Qt versions): walk the tree.
            Err(zbus::Error::MethodError(..)) => self.walk_for_focus(name, window),
            Err(error) => Err(error),
        }
    }

    /// Depth-first through showing objects, within [`WALK_NODES`] and
    /// [`WALK_TIME`]; descendants of a `MANAGES_DESCENDANTS` container
    /// (long lists and tables) are not walked. The whole showing tree is
    /// walked, focused objects' children included, so a second focused
    /// object is seen: then, over budget, or when a call fails (a part of
    /// the tree unseen), there is no answer.
    fn walk_for_focus(&self, name: &str, window: &str) -> zbus::Result<Option<OwnedObjectPath>> {
        let deadline = Instant::now() + WALK_TIME;
        let mut stack: Vec<Object> = self.children(name, window)?;
        let mut visited = 0;
        let mut found = None;
        while let Some((owner, path)) = stack.pop() {
            visited += 1;
            if visited > WALK_NODES || Instant::now() > deadline {
                log::debug!("AT-SPI: focus walk over budget");
                return Ok(None);
            }
            if owner != name {
                continue;
            }
            let states = self.states(&owner, path.as_str())?;
            if !states.has(STATE_SHOWING) || states.has(STATE_MANAGES_DESCENDANTS) {
                if states.has(STATE_FOCUSED) && found.replace(path).is_some() {
                    log::debug!("AT-SPI: more than one focused object; reading none");
                    return Ok(None);
                }
                continue;
            }
            let children = self.children(&owner, path.as_str())?;
            if states.has(STATE_FOCUSED) && found.replace(path).is_some() {
                log::debug!("AT-SPI: more than one focused object; reading none");
                return Ok(None);
            }
            // Reversed, so the first child is walked first.
            stack.extend(children.into_iter().rev());
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn states_round_trip_through_both_words() {
        let states = States::of(&[STATE_FOCUSED, STATE_READ_ONLY]);
        assert_eq!(states.words(), vec![1 << 12, 1 << (43 - 32)]);
        let back = States::from_words(&states.words());
        assert!(back.has(STATE_FOCUSED) && back.has(STATE_READ_ONLY));
        assert!(!back.has(STATE_EDITABLE));
        assert_eq!(States::from_words(&[]), States::default());
    }

    #[test]
    fn only_the_last_characters_are_kept() {
        assert_eq!(last_chars("abc", 2), "bc");
        assert_eq!(last_chars("abc", 3), "abc");
        assert_eq!(last_chars("abc", 9), "abc");
        assert_eq!(last_chars("éü😀x", 2), "😀x");
        assert_eq!(last_chars("abc", 0), "");
        assert_eq!(last_chars("", 4), "");
    }
}
