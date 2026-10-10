//! The portal worker against a scripted GlobalShortcuts portal on a
//! private `dbus-daemon` (skipped where none is installed). The fake
//! follows the xdg-desktop-portal frontend's wire shapes: request and
//! session handles derived from the caller's tokens, `Response` on the
//! request object, `session_handle` as a string, `a(sa{sv})` lists.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zbus::message::Header;
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue, Value};

use super::dbus::{DESTINATION, INTERFACE, PATH};
use super::{APP_ID, PortalShortcuts, PortalStatus};
use crate::activation::{Activation, ActivationConfig, Effect};
use crate::shortcut::{GlobalEvent, Shortcut};
use starling_dictation::settings::ActivationMode;

/// A private session bus, killed on drop.
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
                eprintln!("skipped: no dbus-daemon to run a private bus");
                return;
            }
        }
    };
}

/// What the fake portal does and what it saw.
#[derive(Default)]
struct Script {
    version: u32,
    /// ListShortcuts reports a binding remembered from an earlier run.
    remembered: bool,
    /// BindShortcuts' response code, and the trigger it reports bound.
    bind_response: u32,
    bound_trigger: Option<String>,
    /// Bind succeeds but binds nothing (the user unticked it).
    bind_nothing: bool,
    /// The dialog never answers.
    bind_hangs: bool,
    /// Right after a successful bind, the user removes the shortcut.
    remove_after_bind: bool,
    /// The desktop closes each session right after creating it.
    close_after_create: bool,
    registered: Vec<String>,
    sessions: Vec<String>,
    closed: Vec<String>,
    preferred: Vec<Option<String>>,
    configured: usize,
}

type Shared = Arc<Mutex<Script>>;

fn sender_part(header: &Header<'_>) -> String {
    header
        .sender()
        .map(|sender| sender.trim_start_matches(':').replace('.', "_"))
        .unwrap_or_default()
}

fn token(options: &HashMap<String, OwnedValue>, key: &str) -> String {
    options
        .get(key)
        .and_then(|value| String::try_from(value.clone()).ok())
        .expect(key)
}

async fn respond(
    connection: &zbus::Connection,
    request: &str,
    code: u32,
    results: HashMap<&str, Value<'_>>,
) {
    connection
        .emit_signal(
            None::<&str>,
            request,
            "org.freedesktop.portal.Request",
            "Response",
            &(code, results),
        )
        .await
        .expect("emit Response");
}

type ShortcutList = Vec<(String, HashMap<String, Value<'static>>)>;

/// This app's shortcut as the portal lists it (`a(sa{sv})`).
fn shortcut_entries(trigger: Option<&str>) -> ShortcutList {
    let mut info: HashMap<String, Value<'static>> =
        HashMap::from([("description".to_string(), Value::from("Dictate"))]);
    if let Some(trigger) = trigger {
        info.insert(
            "trigger_description".to_string(),
            Value::from(trigger.to_string()),
        );
    }
    vec![("record".to_string(), info)]
}

/// The same list as a results-dict value.
async fn close(connection: &zbus::Connection, session: &str) {
    connection
        .emit_signal(
            None::<&str>,
            session,
            "org.freedesktop.portal.Session",
            "Closed",
            &(HashMap::<&str, Value>::new(),),
        )
        .await
        .expect("emit Closed");
}

fn shortcut_list(trigger: Option<&str>) -> Value<'static> {
    Value::from(shortcut_entries(trigger))
}

struct FakeGlobalShortcuts(Shared);

#[zbus::interface(name = "org.freedesktop.portal.GlobalShortcuts")]
impl FakeGlobalShortcuts {
    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        self.0.lock().unwrap().version
    }

    async fn create_session(
        &self,
        options: HashMap<String, OwnedValue>,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] connection: &zbus::Connection,
    ) -> OwnedObjectPath {
        let sender = sender_part(&header);
        let request = format!(
            "{PATH}/request/{sender}/{}",
            token(&options, "handle_token")
        );
        let session = format!(
            "{PATH}/session/{sender}/{}",
            token(&options, "session_handle_token")
        );
        connection
            .object_server()
            .at(session.as_str(), FakeSession(self.0.clone()))
            .await
            .expect("export session");
        self.0.lock().unwrap().sessions.push(session.clone());
        // Responds before the method returns: the client must have
        // subscribed before calling.
        respond(
            connection,
            &request,
            0,
            HashMap::from([("session_handle", Value::from(session.clone()))]),
        )
        .await;
        if self.0.lock().unwrap().close_after_create {
            close(connection, &session).await;
        }
        OwnedObjectPath::try_from(request).unwrap()
    }

    async fn bind_shortcuts(
        &self,
        session: ObjectPath<'_>,
        shortcuts: Vec<(String, HashMap<String, OwnedValue>)>,
        _parent_window: String,
        options: HashMap<String, OwnedValue>,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] connection: &zbus::Connection,
    ) -> OwnedObjectPath {
        let request = format!(
            "{PATH}/request/{}/{}",
            sender_part(&header),
            token(&options, "handle_token")
        );
        let preferred = shortcuts
            .iter()
            .find(|(id, _)| id == "record")
            .and_then(|(_, info)| info.get("preferred_trigger"))
            .and_then(|value| String::try_from(value.clone()).ok());
        let (code, trigger, nothing, hangs, remove) = {
            let mut script = self.0.lock().unwrap();
            script.preferred.push(preferred);
            (
                script.bind_response,
                script.bound_trigger.clone(),
                script.bind_nothing,
                script.bind_hangs,
                script.remove_after_bind,
            )
        };
        let session = session.to_string();
        if hangs {
            return OwnedObjectPath::try_from(request).unwrap();
        }
        let connection = connection.clone();
        let reply_path = request.clone();
        // The dialog answers later, after the method returned.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            zbus::block_on(async {
                let results = if code == 0 && !nothing {
                    HashMap::from([("shortcuts", shortcut_list(trigger.as_deref()))])
                } else if code == 0 {
                    HashMap::from([(
                        "shortcuts",
                        Value::from(Vec::<(String, HashMap<String, Value>)>::new()),
                    )])
                } else {
                    HashMap::new()
                };
                respond(&connection, &reply_path, code, results).await;
                if remove {
                    // Back to back with the response.
                    connection
                        .emit_signal(
                            None::<&str>,
                            PATH,
                            INTERFACE,
                            "ShortcutsChanged",
                            &(
                                ObjectPath::try_from(session.as_str()).unwrap(),
                                ShortcutList::new(),
                            ),
                        )
                        .await
                        .unwrap();
                }
            });
        });
        OwnedObjectPath::try_from(request).unwrap()
    }

    async fn list_shortcuts(
        &self,
        _session: ObjectPath<'_>,
        options: HashMap<String, OwnedValue>,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] connection: &zbus::Connection,
    ) -> OwnedObjectPath {
        let request = format!(
            "{PATH}/request/{}/{}",
            sender_part(&header),
            token(&options, "handle_token")
        );
        let remembered = self.0.lock().unwrap().remembered;
        let list = if remembered {
            shortcut_list(Some("Ctrl+Shift+Space"))
        } else {
            Value::from(Vec::<(String, HashMap<String, Value>)>::new())
        };
        respond(
            connection,
            &request,
            0,
            HashMap::from([("shortcuts", list)]),
        )
        .await;
        OwnedObjectPath::try_from(request).unwrap()
    }

    fn configure_shortcuts(
        &self,
        _session: ObjectPath<'_>,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) {
        self.0.lock().unwrap().configured += 1;
    }
}

struct FakeSession(Shared);

#[zbus::interface(name = "org.freedesktop.portal.Session")]
impl FakeSession {
    fn close(&self, #[zbus(header)] header: Header<'_>) {
        let path = header
            .path()
            .map(|path| path.to_string())
            .unwrap_or_default();
        self.0.lock().unwrap().closed.push(path);
    }
}

struct FakeRegistry(Shared);

#[zbus::interface(name = "org.freedesktop.host.portal.Registry")]
impl FakeRegistry {
    fn register(&self, app_id: String, _options: HashMap<String, OwnedValue>) {
        self.0.lock().unwrap().registered.push(app_id);
    }
}

/// The fake portal's own connection, owning the portal's name.
fn fake_portal(bus: &PrivateBus, script: Script) -> (zbus::blocking::Connection, Shared) {
    let shared: Shared = Arc::new(Mutex::new(script));
    let connection = zbus::blocking::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name(DESTINATION)
        .unwrap()
        .serve_at(PATH, FakeGlobalShortcuts(shared.clone()))
        .unwrap()
        .serve_at(PATH, FakeRegistry(shared.clone()))
        .unwrap()
        .build()
        .expect("fake portal");
    (connection, shared)
}

fn bound_script() -> Script {
    Script {
        version: 2,
        bound_trigger: Some("Ctrl+Shift+Space".to_string()),
        ..Default::default()
    }
}

fn emit(portal: &zbus::blocking::Connection, signal: &str, session: &str, id: &str) {
    portal
        .emit_signal(
            None::<&str>,
            PATH,
            INTERFACE,
            signal,
            &(
                ObjectPath::try_from(session).unwrap(),
                id,
                7u64,
                HashMap::<&str, Value>::new(),
            ),
        )
        .unwrap();
}

/// Poll the handle until `done` holds for its status, collecting the
/// machine inputs that arrived meanwhile.
fn wait_for(
    client: &mut PortalShortcuts,
    events: &mut Vec<GlobalEvent>,
    what: &str,
    done: impl Fn(&PortalStatus, &[GlobalEvent]) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        while let Some(event) = client.next_event() {
            events.push(event);
        }
        if done(client.status(), events) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}; status {:?}, events {events:?}",
            client.status()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn shortcut(text: &str) -> Shortcut {
    Shortcut::parse(text).unwrap()
}

#[test]
fn no_portal_on_the_bus_is_reported_unavailable() {
    let bus = private_bus!();
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "unavailable", |status, _| {
        matches!(status, PortalStatus::Unavailable(_))
    });
    let PortalStatus::Unavailable(reason) = client.status() else {
        unreachable!()
    };
    assert!(reason.contains("not running"), "{reason}");
    assert!(!client.is_bound());
}

#[test]
fn a_portal_without_global_shortcuts_is_reported_unavailable() {
    let bus = private_bus!();
    // Owns the portal name but exports only the registry.
    let shared: Shared = Arc::new(Mutex::new(Script::default()));
    let _portal = zbus::blocking::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name(DESTINATION)
        .unwrap()
        .serve_at(PATH, FakeRegistry(shared))
        .unwrap()
        .build()
        .unwrap();
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "unavailable", |status, _| {
        matches!(status, PortalStatus::Unavailable(_))
    });
    let PortalStatus::Unavailable(reason) = client.status() else {
        unreachable!()
    };
    assert!(
        reason.contains("does not implement GlobalShortcuts"),
        "{reason}"
    );
}

#[test]
fn a_first_run_waits_for_set_up_then_delivers_hold_to_talk_edges() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(&bus, bound_script());
    let mut client = PortalShortcuts::spawn(
        Some(bus.address.clone()),
        None,
        &shortcut("Ctrl+Shift+Space"),
    );
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        matches!(status, PortalStatus::NeedsSetup { configurable: true })
    });
    // Nothing is bound (no dialog) until the user asks.
    assert!(script.lock().unwrap().preferred.is_empty());
    assert_eq!(script.lock().unwrap().registered, vec![APP_ID.to_string()]);

    client.set_up(&shortcut("Ctrl+Shift+Space"));
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    assert_eq!(
        client.status(),
        &PortalStatus::Bound {
            trigger: Some("Ctrl+Shift+Space".to_string()),
            configurable: true
        }
    );
    assert_eq!(
        script.lock().unwrap().preferred,
        vec![Some("CTRL+SHIFT+space".to_string())]
    );
    let session = script.lock().unwrap().sessions[0].clone();

    // Another app's shortcut or another session never reaches the machine.
    emit(&portal, "Activated", &session, "other");
    emit(
        &portal,
        "Activated",
        &format!("{PATH}/session/x/y"),
        "record",
    );
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    emit(&portal, "Deactivated", &session, "record");
    wait_for(&mut client, &mut events, "release", |_, events| {
        events.len() >= 2
    });
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );

    // Through the machine: one hold-to-talk take.
    let mut machine = Activation::new(ActivationConfig {
        mode: ActivationMode::Hold,
        double_tap_hands_free: false,
    });
    let GlobalEvent::Pressed(down) = events[0] else {
        unreachable!()
    };
    let [Effect::Start(take)] = machine.press(down, true)[..] else {
        panic!("no start")
    };
    machine.samples_arrived(take);
    // The release is taken as long after the press as the test needs.
    assert_eq!(
        machine.release(down + Duration::from_secs(2)),
        vec![Effect::Finish(take)]
    );

    client.configure();
    wait_for(&mut client, &mut events, "configure call", |_, _| {
        script.lock().unwrap().configured == 1
    });
}

#[test]
fn a_remembered_binding_is_bound_again_at_start() {
    let bus = private_bus!();
    let (_portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    assert_eq!(
        script.lock().unwrap().preferred,
        vec![Some("F9".to_string())]
    );
}

#[test]
fn a_declined_dialog_can_be_retried_in_a_fresh_session() {
    let bus = private_bus!();
    let (_portal, script) = fake_portal(
        &bus,
        Script {
            bind_response: 1,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        matches!(status, PortalStatus::NeedsSetup { .. })
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "declined", |status, _| {
        matches!(status, PortalStatus::Declined { .. })
    });
    script.lock().unwrap().bind_response = 0;
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let script = script.lock().unwrap();
    // The portal binds once per session: the retry opened a second one
    // and closed the first.
    assert_eq!(script.sessions.len(), 2);
    assert_eq!(script.closed, vec![script.sessions[0].clone()]);
    assert!(events.is_empty(), "no key edges: {events:?}");
}

#[test]
fn binding_nothing_reads_as_declined() {
    let bus = private_bus!();
    let (_portal, _script) = fake_portal(
        &bus,
        Script {
            bind_nothing: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "declined", |status, _| {
        matches!(status, PortalStatus::Declined { .. })
    });
}

#[test]
fn a_session_closed_mid_hold_releases_the_shortcut() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    portal
        .emit_signal(
            None::<&str>,
            session.as_str(),
            "org.freedesktop.portal.Session",
            "Closed",
            &(HashMap::<&str, Value>::new(),),
        )
        .unwrap();
    wait_for(&mut client, &mut events, "lost", |status, _| {
        matches!(status, PortalStatus::Lost { .. })
    });
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
    // Edges from the closed session are ignored.
    emit(&portal, "Deactivated", &session, "record");
    emit(&portal, "Activated", &session, "record");
    std::thread::sleep(Duration::from_millis(200));
    while let Some(event) = client.next_event() {
        events.push(event);
    }
    assert_eq!(events.len(), 2, "{events:?}");
}

#[test]
fn shortcut_changes_follow_the_desktop() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    let changed = |trigger: Option<&str>| {
        let list = match trigger {
            Some(trigger) => shortcut_entries(Some(trigger)),
            None => ShortcutList::new(),
        };
        portal
            .emit_signal(
                None::<&str>,
                PATH,
                INTERFACE,
                "ShortcutsChanged",
                &(ObjectPath::try_from(session.as_str()).unwrap(), list),
            )
            .unwrap();
    };
    changed(Some("Meta+D"));
    wait_for(
        &mut client,
        &mut events,
        "new trigger",
        |status, _| matches!(status, PortalStatus::Bound { trigger: Some(t), .. } if t == "Meta+D"),
    );
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    // Removed in the desktop's settings while held: released, and lost.
    changed(None);
    wait_for(&mut client, &mut events, "removed", |status, _| {
        matches!(status, PortalStatus::Lost { .. })
    });
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
}

#[test]
fn a_new_shortcut_is_offered_through_a_fresh_session() {
    let bus = private_bus!();
    let (_portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    // Unchanged keys: nothing happens.
    client.rebind(&shortcut("F9"));
    client.rebind(&shortcut("Alt+D"));
    wait_for(&mut client, &mut events, "second bind", |status, _| {
        status.is_bound() && script.lock().unwrap().preferred.len() == 2
    });
    let script = script.lock().unwrap();
    assert_eq!(
        script.preferred,
        vec![Some("F9".to_string()), Some("ALT+d".to_string())]
    );
    assert_eq!(script.sessions.len(), 2);
    assert_eq!(script.closed, vec![script.sessions[0].clone()]);
}

#[test]
fn the_portal_going_away_releases_and_reports_it() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    drop(portal);
    wait_for(
        &mut client,
        &mut events,
        "lost",
        |status, _| matches!(status, PortalStatus::Lost { reason, .. } if reason.contains("portal stopped")),
    );
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
}

#[test]
fn a_press_and_release_sent_back_to_back_keep_their_order() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    // A quick tap: both edges are on the wire before either is handled.
    for _ in 0..20 {
        emit(&portal, "Activated", &session, "record");
        emit(&portal, "Deactivated", &session, "record");
    }
    wait_for(&mut client, &mut events, "40 edges", |_, events| {
        events.len() >= 40
    });
    for pair in events.chunks(2) {
        assert!(
            matches!(pair, [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]),
            "{events:?}"
        );
    }
}

#[test]
fn the_portal_vanishing_during_the_dialog_ends_the_wait() {
    let bus = private_bus!();
    let (portal, _script) = fake_portal(
        &bus,
        Script {
            bind_hangs: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "binding", |status, _| {
        *status == PortalStatus::Binding
    });
    drop(portal);
    wait_for(
        &mut client,
        &mut events,
        "lost",
        |status, _| matches!(status, PortalStatus::Lost { reason, .. } if reason.contains("portal stopped")),
    );
}

#[test]
fn a_restarted_portal_is_registered_with_again_and_rebinds() {
    let bus = private_bus!();
    let remembered = || Script {
        remembered: true,
        ..bound_script()
    };
    let (portal, script) = fake_portal(&bus, remembered());
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    drop(portal);
    wait_for(&mut client, &mut events, "lost", |status, _| {
        matches!(status, PortalStatus::Lost { .. })
    });
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
    // A new instance: the worker reconnects, registers first, and binds
    // the shortcut the desktop remembers.
    let (portal, script) = fake_portal(&bus, remembered());
    wait_for(&mut client, &mut events, "bound again", |status, _| {
        status.is_bound()
    });
    assert_eq!(script.lock().unwrap().registered, vec![APP_ID.to_string()]);
    let session = script.lock().unwrap().sessions[0].clone();
    emit(&portal, "Activated", &session, "record");
    wait_for(
        &mut client,
        &mut events,
        "press on the new instance",
        |_, events| events.len() >= 3,
    );
}

#[test]
fn losing_the_bus_mid_hold_releases_and_reports_it() {
    let mut bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    let session = script.lock().unwrap().sessions[0].clone();
    emit(&portal, "Activated", &session, "record");
    wait_for(&mut client, &mut events, "press", |_, events| {
        !events.is_empty()
    });
    let _ = bus.child.kill();
    let _ = bus.child.wait();
    wait_for(&mut client, &mut events, "unavailable", |status, _| {
        matches!(status, PortalStatus::Unavailable(_))
    });
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
}

#[test]
fn a_session_closed_while_the_dialog_is_open_ends_the_wait() {
    let bus = private_bus!();
    let (portal, script) = fake_portal(
        &bus,
        Script {
            bind_hangs: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "binding", |status, _| {
        *status == PortalStatus::Binding
    });
    let session = script.lock().unwrap().sessions[0].clone();
    portal
        .emit_signal(
            None::<&str>,
            session.as_str(),
            "org.freedesktop.portal.Session",
            "Closed",
            &(HashMap::<&str, Value>::new(),),
        )
        .unwrap();
    wait_for(
        &mut client,
        &mut events,
        "lost",
        |status, _| matches!(status, PortalStatus::Lost { reason, .. } if reason.contains("closed")),
    );
    // The worker is free again: a new set-up binds in a new session.
    script.lock().unwrap().bind_hangs = false;
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    assert_eq!(script.lock().unwrap().sessions.len(), 2);
}

#[test]
fn a_removal_right_after_the_bind_response_is_not_lost() {
    let bus = private_bus!();
    let (_portal, _script) = fake_portal(
        &bus,
        Script {
            remove_after_bind: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(
        &mut client,
        &mut events,
        "removed",
        |status, _| matches!(status, PortalStatus::Lost { reason, .. } if reason.contains("removed")),
    );
    std::thread::sleep(Duration::from_millis(100));
    while client.next_event().is_some() {}
    assert!(!client.is_bound(), "{:?}", client.status());
}

#[test]
fn a_session_closed_right_after_creation_is_not_used() {
    let bus = private_bus!();
    let (_portal, script) = fake_portal(
        &bus,
        Script {
            close_after_create: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    // Whether `Closed` is handled before or after CreateSession's caller
    // looks, the worker waits for set-up with no session; it never lists
    // or binds on the closed one.
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        matches!(status, PortalStatus::NeedsSetup { .. })
    });
    std::thread::sleep(Duration::from_millis(100));
    while client.next_event().is_some() {}
    assert!(
        matches!(client.status(), PortalStatus::NeedsSetup { .. }),
        "{:?}",
        client.status()
    );
    assert!(script.lock().unwrap().preferred.is_empty());
    script.lock().unwrap().close_after_create = false;
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    // The bind went to a fresh session, not the closed one.
    assert_eq!(script.lock().unwrap().sessions.len(), 2);
}

#[test]
fn losing_the_bus_during_the_dialog_leaves_it_unavailable() {
    let mut bus = private_bus!();
    let (_portal, _script) = fake_portal(
        &bus,
        Script {
            bind_hangs: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "binding", |status, _| {
        *status == PortalStatus::Binding
    });
    let _ = bus.child.kill();
    let _ = bus.child.wait();
    wait_for(&mut client, &mut events, "unavailable", |status, _| {
        matches!(status, PortalStatus::Unavailable(_))
    });
    std::thread::sleep(Duration::from_millis(200));
    while client.next_event().is_some() {}
    // Nothing overwrote it with a state that offers set-up again.
    assert!(
        matches!(client.status(), PortalStatus::Unavailable(_)),
        "{:?}",
        client.status()
    );
}

#[test]
fn a_portal_replaced_outright_during_the_dialog_starts_over() {
    let bus = private_bus!();
    let (_old, _old_script) = fake_portal(
        &bus,
        Script {
            bind_hangs: true,
            ..bound_script()
        },
    );
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "needs setup", |status, _| {
        status.can_set_up()
    });
    client.set_up(&shortcut("F9"));
    wait_for(&mut client, &mut events, "binding", |status, _| {
        *status == PortalStatus::Binding
    });
    // `xdg-desktop-portal --replace`: the name moves to a new owner with
    // no gap (zbus's default flags allow and request replacement).
    let (_new, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    wait_for(
        &mut client,
        &mut events,
        "bound by the new instance",
        |status, _| status.is_bound(),
    );
    assert_eq!(script.lock().unwrap().registered, vec![APP_ID.to_string()]);
}

#[test]
fn a_portal_that_appears_after_startup_is_picked_up() {
    let bus = private_bus!();
    let mut client = PortalShortcuts::spawn(Some(bus.address.clone()), None, &shortcut("F9"));
    let mut events = Vec::new();
    wait_for(&mut client, &mut events, "unavailable", |status, _| {
        matches!(status, PortalStatus::Unavailable(_))
    });
    let (_portal, script) = fake_portal(
        &bus,
        Script {
            remembered: true,
            ..bound_script()
        },
    );
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    assert_eq!(script.lock().unwrap().registered, vec![APP_ID.to_string()]);
}

// ---- Through the real xdg-desktop-portal frontend ---------------------------
//
// The installed frontend (`/usr/lib/xdg-desktop-portal`, or
// `STARLING_XDP_FRONTEND`) on the private bus, with a scripted *backend*
// (`org.freedesktop.impl.portal.GlobalShortcuts`) chosen through the
// frontend's test hook `XDG_DESKTOP_PORTAL_DIR`. This checks the worker
// against the frontend's own rules: the host-app registry before any
// other call, the app id it then requires, request/session handles,
// unicast signal delivery. Opt-in (`--ignored`): it needs the frontend
// installed and starts a real process.

const IMPL_NAME: &str = "org.freedesktop.impl.portal.desktop.starlingtest";

#[derive(Default)]
struct ImplScript {
    app_ids: Vec<String>,
    sessions: Vec<String>,
    preferred: Vec<Option<String>>,
}

struct ImplShortcuts(Arc<Mutex<ImplScript>>);

fn owned(value: Value<'_>) -> OwnedValue {
    OwnedValue::try_from(value).expect("owned value")
}

#[zbus::interface(name = "org.freedesktop.impl.portal.GlobalShortcuts")]
impl ImplShortcuts {
    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }

    async fn create_session(
        &self,
        _handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        _options: HashMap<String, OwnedValue>,
        #[zbus(connection)] connection: &zbus::Connection,
    ) -> (u32, HashMap<String, OwnedValue>) {
        connection
            .object_server()
            .at(session_handle.as_str(), ImplSession)
            .await
            .expect("export impl session");
        let mut script = self.0.lock().unwrap();
        script.app_ids.push(app_id);
        script.sessions.push(session_handle.to_string());
        (0, HashMap::new())
    }

    async fn bind_shortcuts(
        &self,
        _handle: OwnedObjectPath,
        _session_handle: OwnedObjectPath,
        shortcuts: Vec<(String, HashMap<String, OwnedValue>)>,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) -> (u32, HashMap<String, OwnedValue>) {
        let preferred = shortcuts
            .iter()
            .find(|(id, _)| id == "record")
            .and_then(|(_, info)| info.get("preferred_trigger"))
            .and_then(|value| String::try_from(value.clone()).ok());
        self.0.lock().unwrap().preferred.push(preferred);
        (
            0,
            HashMap::from([(
                "shortcuts".to_string(),
                owned(shortcut_list(Some("Ctrl+Shift+Space"))),
            )]),
        )
    }

    async fn list_shortcuts(
        &self,
        _handle: OwnedObjectPath,
        _session_handle: OwnedObjectPath,
    ) -> (u32, HashMap<String, OwnedValue>) {
        let empty = Value::from(Vec::<(String, HashMap<String, Value>)>::new());
        (0, HashMap::from([("shortcuts".to_string(), owned(empty))]))
    }

    fn configure_shortcuts(
        &self,
        _session_handle: OwnedObjectPath,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) {
    }
}

struct ImplSession;

#[zbus::interface(name = "org.freedesktop.impl.portal.Session")]
impl ImplSession {
    fn close(&self) {}

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}

/// The frontend process, killed on drop.
struct Frontend(Child);

impl Drop for Frontend {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "starts the installed xdg-desktop-portal frontend on a private bus"]
fn the_real_portal_frontend_binds_and_forwards_hold_edges() {
    let frontend_path = std::env::var("STARLING_XDP_FRONTEND")
        .unwrap_or_else(|_| "/usr/lib/xdg-desktop-portal".to_string());
    if !std::path::Path::new(&frontend_path).exists() {
        eprintln!("skipped: no xdg-desktop-portal at {frontend_path}");
        return;
    }
    let bus = private_bus!();
    let portals = tempfile::tempdir().unwrap();
    std::fs::write(
        portals.path().join("starlingtest.portal"),
        format!(
            "[portal]\nDBusName={IMPL_NAME}\n\
             Interfaces=org.freedesktop.impl.portal.GlobalShortcuts;\nUseIn=starlingtest\n"
        ),
    )
    .unwrap();
    let script = Arc::new(Mutex::new(ImplScript::default()));
    let backend = zbus::blocking::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name(IMPL_NAME)
        .unwrap()
        .serve_at(PATH, ImplShortcuts(script.clone()))
        .unwrap()
        .build()
        .expect("scripted backend");
    let scratch = tempfile::tempdir().unwrap();
    let _frontend = Frontend(
        Command::new(&frontend_path)
            .env_clear()
            .env("DBUS_SESSION_BUS_ADDRESS", &bus.address)
            .env("XDG_DESKTOP_PORTAL_DIR", portals.path())
            .env("XDG_CURRENT_DESKTOP", "starlingtest")
            .env("XDG_RUNTIME_DIR", scratch.path())
            .env("XDG_DATA_HOME", scratch.path())
            .env("XDG_CONFIG_HOME", scratch.path())
            .env("HOME", scratch.path())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start xdg-desktop-portal"),
    );
    // Wait for the frontend to own its name.
    let probe = zbus::blocking::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .build()
        .unwrap();
    let dbus = zbus::blocking::fdo::DBusProxy::new(&probe).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while !dbus
        .name_has_owner(DESTINATION.try_into().unwrap())
        .unwrap_or(false)
    {
        assert!(
            Instant::now() < deadline,
            "the frontend never took {DESTINATION}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // The frontend resolves a registered host app id through
    // `<id>.desktop` in its data dirs; the scratch data home has none yet.
    let entries = scratch.path().join("applications");
    let mut client = PortalShortcuts::spawn(
        Some(bus.address.clone()),
        Some(entries.clone()),
        &shortcut("Ctrl+Shift+Space"),
    );
    let mut events = Vec::new();
    wait_for(
        &mut client,
        &mut events,
        "needs a desktop entry",
        |status, _| *status == PortalStatus::NeedsDesktopEntry,
    );
    assert!(script.lock().unwrap().app_ids.is_empty());
    // Set-up installs the entry, reconnects, registers and binds.
    client.set_up(&shortcut("Ctrl+Shift+Space"));
    wait_for(&mut client, &mut events, "bound", |status, _| {
        status.is_bound()
    });
    assert!(entries.join(format!("{APP_ID}.desktop")).is_file());
    // The frontend accepted the session only with an app id: the one the
    // worker registered.
    assert_eq!(script.lock().unwrap().app_ids, vec![APP_ID.to_string()]);
    assert_eq!(
        script.lock().unwrap().preferred,
        vec![Some("CTRL+SHIFT+space".to_string())]
    );
    let session = script.lock().unwrap().sessions[0].clone();
    for signal in ["Activated", "Deactivated"] {
        backend
            .emit_signal(
                None::<&str>,
                PATH,
                "org.freedesktop.impl.portal.GlobalShortcuts",
                signal,
                &(
                    ObjectPath::try_from(session.as_str()).unwrap(),
                    "record",
                    7u64,
                    HashMap::<&str, Value>::new(),
                ),
            )
            .unwrap();
    }
    wait_for(
        &mut client,
        &mut events,
        "press and release",
        |_, events| events.len() >= 2,
    );
    assert!(
        matches!(
            events[..],
            [GlobalEvent::Pressed(_), GlobalEvent::Released(_)]
        ),
        "{events:?}"
    );
}
