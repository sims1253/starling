//! The AT-SPI field reader against scripted applications on a private
//! `dbus-daemon` (skipped where none is installed): a registry root
//! listing the apps, each with a window, a panel and text fields that
//! follow AT-SPI's wire shapes (`a(so)` children, `au` states, the Text
//! interface's `CaretOffset`/`GetText`/selections, `Collection.GetMatches`
//! where the scripted toolkit implements it).
#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use starling_insertion::atspi::{
    AtspiReader, States, ROLE_PASSWORD_TEXT, STATE_ACTIVE, STATE_EDITABLE, STATE_FOCUSED,
    STATE_READ_ONLY, STATE_SHOWING,
};
use starling_insertion::{BackendKind, FieldReader, Surrounding, SurroundingText, TargetSnapshot};
use zbus::blocking::Connection;
use zbus::zvariant::OwnedObjectPath;

const ROOT: &str = "/org/a11y/atspi/accessible/root";
const ROLE_ENTRY: u32 = 79;

/// A private bus, killed on drop.
struct PrivateBus {
    child: Child,
    address: String,
    _dir: tempfile::TempDir,
}

impl PrivateBus {
    fn start() -> Option<PrivateBus> {
        let dir = tempfile::tempdir().ok()?;
        let config = dir.path().join("bus.conf");
        std::fs::write(
            &config,
            format!(
                r#"<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>unix:dir={}</listen>
  <auth>EXTERNAL</auth>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>"#,
                dir.path().display()
            ),
        )
        .ok()?;
        let mut child = Command::new("dbus-daemon")
            .arg(format!("--config-file={}", config.display()))
            .args(["--nofork", "--print-address=1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut line = String::new();
        BufReader::new(child.stdout.take()?)
            .read_line(&mut line)
            .ok()?;
        Some(PrivateBus {
            child,
            address: line.trim().to_string(),
            _dir: dir,
        })
    }

    fn connect(&self) -> Connection {
        zbus::blocking::connection::Builder::address(self.address.as_str())
            .unwrap()
            .build()
            .unwrap()
    }
}

impl Drop for PrivateBus {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

macro_rules! private_bus {
    () => {
        match PrivateBus::start() {
            Some(bus) => bus,
            None => {
                eprintln!("skipped: no dbus-daemon");
                return;
            }
        }
    };
}

type Children = Vec<(String, OwnedObjectPath)>;

/// One object's `org.a11y.atspi.Accessible`.
struct Node {
    role: u32,
    states: Arc<Mutex<Vec<u32>>>,
    children: Arc<Mutex<Children>>,
}

#[zbus::interface(name = "org.a11y.atspi.Accessible")]
impl Node {
    fn get_children(&self) -> Children {
        self.children.lock().unwrap().clone()
    }

    fn get_state(&self) -> Vec<u32> {
        States::of(&self.states.lock().unwrap()).words()
    }

    fn get_role(&self) -> u32 {
        self.role
    }
}

/// A window's `org.a11y.atspi.Collection`: the focused fields under it.
struct Collection {
    fields: Vec<(OwnedObjectPath, Arc<Mutex<Vec<u32>>>)>,
    owner: String,
}

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

#[zbus::interface(name = "org.a11y.atspi.Collection")]
impl Collection {
    fn get_matches(&self, rule: MatchRule, _sortby: u32, _count: i32, _traverse: bool) -> Children {
        let wanted: Vec<u32> = rule.0.iter().map(|word| *word as u32).collect();
        assert_eq!(wanted, States::of(&[STATE_FOCUSED]).words());
        self.fields
            .iter()
            .filter(|(_, states)| states.lock().unwrap().contains(&STATE_FOCUSED))
            .map(|(path, _)| (self.owner.clone(), path.clone()))
            .collect()
    }
}

/// A field's `org.a11y.atspi.Text`; counts every call that returns text.
struct Text {
    content: Arc<Mutex<String>>,
    caret: Arc<Mutex<i32>>,
    selection: Arc<Mutex<Option<(i32, i32)>>>,
    text_calls: Arc<AtomicUsize>,
}

#[zbus::interface(name = "org.a11y.atspi.Text")]
impl Text {
    #[zbus(property)]
    fn caret_offset(&self) -> i32 {
        *self.caret.lock().unwrap()
    }

    fn get_n_selections(&self) -> i32 {
        i32::from(self.selection.lock().unwrap().is_some())
    }

    fn get_selection(&self, _index: i32) -> (i32, i32) {
        self.selection.lock().unwrap().unwrap_or((0, 0))
    }

    fn get_text(&self, start: i32, end: i32) -> String {
        self.text_calls.fetch_add(1, Ordering::SeqCst);
        let content = self.content.lock().unwrap();
        let end = if end < 0 { i32::MAX } else { end };
        content
            .chars()
            .skip(start.max(0) as usize)
            .take((end - start.max(0)).max(0) as usize)
            .collect()
    }
}

/// One scripted field, for the test to change.
struct Field {
    path: String,
    states: Arc<Mutex<Vec<u32>>>,
    content: Arc<Mutex<String>>,
    caret: Arc<Mutex<i32>>,
    selection: Arc<Mutex<Option<(i32, i32)>>>,
    text_calls: Arc<AtomicUsize>,
}

impl Field {
    fn focus(&self, focused: bool) {
        let mut states = self.states.lock().unwrap();
        states.retain(|state| *state != STATE_FOCUSED);
        if focused {
            states.push(STATE_FOCUSED);
        }
    }

    fn set_text(&self, text: &str, caret: i32) {
        *self.content.lock().unwrap() = text.to_string();
        *self.caret.lock().unwrap() = caret;
    }
}

/// A scripted application: root → window → panel → fields.
struct App {
    connection: Connection,
    window_states: Arc<Mutex<Vec<u32>>>,
    fields: Vec<Field>,
}

impl App {
    fn name(&self) -> String {
        self.connection.unique_name().unwrap().to_string()
    }
}

/// `roles`: one field per role, the first focused. `collection`: whether
/// the window implements `Collection` (GTK 3 does, GTK 4 does not).
fn app(bus: &PrivateBus, roles: &[u32], collection: bool) -> App {
    let connection = bus.connect();
    let name = connection.unique_name().unwrap().to_string();
    let server = connection.object_server();
    let path = |p: &str| OwnedObjectPath::try_from(p.to_string()).unwrap();
    let shared = |v: Vec<u32>| Arc::new(Mutex::new(v));

    let mut fields = Vec::new();
    for (index, role) in roles.iter().enumerate() {
        let field_path = format!("/org/a11y/atspi/accessible/field{index}");
        let mut states = vec![STATE_SHOWING, STATE_EDITABLE];
        if index == 0 {
            states.push(STATE_FOCUSED);
        }
        let field = Field {
            path: field_path.clone(),
            states: shared(states),
            content: Arc::new(Mutex::new(String::new())),
            caret: Arc::new(Mutex::new(0)),
            selection: Arc::new(Mutex::new(None)),
            text_calls: Arc::new(AtomicUsize::new(0)),
        };
        server
            .at(
                field_path.as_str(),
                Node {
                    role: *role,
                    states: field.states.clone(),
                    children: Arc::default(),
                },
            )
            .unwrap();
        server
            .at(
                field_path.as_str(),
                Text {
                    content: field.content.clone(),
                    caret: field.caret.clone(),
                    selection: field.selection.clone(),
                    text_calls: field.text_calls.clone(),
                },
            )
            .unwrap();
        fields.push(field);
    }

    let panel = "/org/a11y/atspi/accessible/panel";
    let field_children: Children = fields
        .iter()
        .map(|field| (name.clone(), path(&field.path)))
        .collect();
    server
        .at(
            panel,
            Node {
                role: 39,
                states: shared(vec![STATE_SHOWING]),
                children: Arc::new(Mutex::new(field_children)),
            },
        )
        .unwrap();

    let window = "/org/a11y/atspi/accessible/window";
    let window_states = shared(vec![STATE_SHOWING, STATE_ACTIVE]);
    server
        .at(
            window,
            Node {
                role: 23,
                states: window_states.clone(),
                children: Arc::new(Mutex::new(vec![(name.clone(), path(panel))])),
            },
        )
        .unwrap();
    if collection {
        server
            .at(
                window,
                Collection {
                    fields: fields
                        .iter()
                        .map(|field| (path(&field.path), field.states.clone()))
                        .collect(),
                    owner: name.clone(),
                },
            )
            .unwrap();
    }
    server
        .at(
            ROOT,
            Node {
                role: 75,
                states: shared(Vec::new()),
                children: Arc::new(Mutex::new(vec![(name.clone(), path(window))])),
            },
        )
        .unwrap();
    drop(server);
    App {
        connection,
        window_states,
        fields,
    }
}

/// The registry root listing `apps`.
fn registry(bus: &PrivateBus, apps: &[&App]) -> Connection {
    let children: Children = apps
        .iter()
        .map(|app| (app.name(), OwnedObjectPath::try_from(ROOT).unwrap()))
        .collect();
    zbus::blocking::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name("org.a11y.atspi.Registry")
        .unwrap()
        .serve_at(
            ROOT,
            Node {
                role: 14,
                states: Arc::default(),
                children: Arc::new(Mutex::new(children)),
            },
        )
        .unwrap()
        .build()
        .unwrap()
}

/// A target of this process (every scripted app runs in it), or without a
/// pid (Wayland).
fn target(pid: Option<u32>) -> TargetSnapshot {
    TargetSnapshot {
        backend: BackendKind::Fake,
        target_ref: "fake:1:1".to_string(),
        app: None,
        title: None,
        pid,
    }
}

fn reader(bus: &PrivateBus) -> AtspiReader {
    AtspiReader::with_address(bus.address.clone(), Vec::new())
}

fn before(text: &str) -> Surrounding {
    Surrounding::Text(SurroundingText {
        before: text.to_string(),
        after: String::new(),
        selection: None,
    })
}

#[test]
fn reads_up_to_128_characters_before_the_caret_of_the_located_field() {
    let bus = private_bus!();
    let gtk3 = app(&bus, &[ROLE_ENTRY], true);
    let _registry = registry(&bus, &[&gtk3]);
    let reader = reader(&bus);
    let here = target(Some(std::process::id()));

    let field = reader.locate(&here).expect("the focused field");
    assert_eq!(field.bus_name, gtk3.name());
    assert_eq!(field.path, gtk3.fields[0].path);

    gtk3.fields[0].set_text("Meet me at noon", 15);
    assert_eq!(reader.read(&here, &field), before("Meet me at noon"));
    // The caret in the middle: only what is before it.
    gtk3.fields[0].set_text("Meet me at noon", 7);
    assert_eq!(reader.read(&here, &field), before("Meet me"));
    // A selection is replaced by the text: its start is the boundary.
    *gtk3.fields[0].selection.lock().unwrap() = Some((10, 3));
    assert_eq!(reader.read(&here, &field), before("Mee"));
    *gtk3.fields[0].selection.lock().unwrap() = None;

    let long: String = "ä".repeat(200) + "Ende";
    gtk3.fields[0].set_text(&long, 204);
    let Surrounding::Text(read) = reader.read(&here, &field) else {
        panic!("no text");
    };
    assert_eq!(read.before.chars().count(), 128);
    assert!(read.before.ends_with("äEnde"));
    assert_eq!(read.after, "", "the text after the caret is never read");
}

#[test]
fn a_toolkit_without_collection_is_walked() {
    let bus = private_bus!();
    let gtk4 = app(&bus, &[ROLE_ENTRY, ROLE_ENTRY], false);
    gtk4.fields[0].focus(false);
    gtk4.fields[1].focus(true);
    let _registry = registry(&bus, &[&gtk4]);
    let reader = reader(&bus);
    let field = reader
        .locate(&target(Some(std::process::id())))
        .expect("the focused field");
    assert_eq!(field.path, gtk4.fields[1].path);
}

#[test]
fn a_password_field_is_protected_and_never_read() {
    let bus = private_bus!();
    let app = app(&bus, &[ROLE_PASSWORD_TEXT], true);
    app.fields[0].set_text("hunter2", 7);
    let _registry = registry(&bus, &[&app]);
    let reader = reader(&bus);
    let here = target(Some(std::process::id()));
    let field = reader.locate(&here).expect("the focused field");
    assert_eq!(reader.read(&here, &field), Surrounding::Protected);
    assert_eq!(app.fields[0].text_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn only_the_field_located_at_capture_is_read_while_it_keeps_focus() {
    let bus = private_bus!();
    let app = app(&bus, &[ROLE_ENTRY, ROLE_ENTRY], true);
    app.fields[0].set_text("first", 5);
    app.fields[1].set_text("second", 6);
    let _registry = registry(&bus, &[&app]);
    let reader = reader(&bus);
    let here = target(Some(std::process::id()));
    let field = reader.locate(&here).expect("the focused field");

    // Focus moved to the other field of the same window.
    app.fields[0].focus(false);
    app.fields[1].focus(true);
    assert_eq!(reader.read(&here, &field), Surrounding::Unsupported);
    // A read-only field (or one no longer editable) is not a target.
    app.fields[0].focus(true);
    app.fields[1].focus(false);
    app.fields[0].states.lock().unwrap().push(STATE_READ_ONLY);
    assert_eq!(reader.read(&here, &field), Surrounding::Unsupported);
    app.fields[0]
        .states
        .lock()
        .unwrap()
        .retain(|s| *s != STATE_READ_ONLY);
    assert_eq!(reader.read(&here, &field), before("first"));
    assert_eq!(app.fields[1].text_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn another_process_or_an_excluded_one_is_never_located_or_read() {
    let bus = private_bus!();
    let app = app(&bus, &[ROLE_ENTRY], true);
    app.fields[0].set_text("private", 7);
    let _registry = registry(&bus, &[&app]);
    let here = target(Some(std::process::id()));
    let field = reader(&bus).locate(&here).expect("the focused field");

    let elsewhere = target(Some(std::process::id() + 1));
    assert_eq!(reader(&bus).locate(&elsewhere), None);
    assert_eq!(
        reader(&bus).read(&elsewhere, &field),
        Surrounding::Unsupported
    );

    let excluding = AtspiReader::with_address(bus.address.clone(), vec![std::process::id()]);
    assert_eq!(excluding.locate(&here), None);
    assert_eq!(app.fields[0].text_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn without_a_pid_only_a_single_active_app_counts() {
    let bus = private_bus!();
    let first = app(&bus, &[ROLE_ENTRY], true);
    let second = app(&bus, &[ROLE_ENTRY], false);
    let _registry = registry(&bus, &[&first, &second]);
    let wayland = target(None);
    // Two apps claim an active window with a focused field: neither.
    assert_eq!(reader(&bus).locate(&wayland), None);

    second
        .window_states
        .lock()
        .unwrap()
        .retain(|s| *s != STATE_ACTIVE);
    let field = reader(&bus)
        .locate(&wayland)
        .expect("the active app's field");
    assert_eq!(field.bus_name, first.name());
    first.fields[0].set_text("Hallo", 5);
    assert_eq!(reader(&bus).read(&wayland, &field), before("Hallo"));
}

#[test]
fn an_unreachable_bus_reads_as_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let address = format!("unix:path={}", dir.path().join("none").display());
    let reader = AtspiReader::with_address(address, Vec::new());
    let here = target(Some(std::process::id()));
    assert_eq!(reader.locate(&here), None);
    let field = starling_insertion::FieldAnchor {
        bus_name: ":1.1".to_string(),
        path: "/x".to_string(),
        pid: std::process::id(),
    };
    assert_eq!(reader.read(&here, &field), Surrounding::Unsupported);
}
