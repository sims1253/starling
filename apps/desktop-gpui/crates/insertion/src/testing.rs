//! The scripted [`FakeBackend`] (feature `test-doubles`): focus,
//! identity and failures under test control, with no X server or
//! Windows session.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    format_ref, insertion_guards, merge_excluded_pids, BackendKind, InsertError, InsertReceipt,
    InsertionBackend, TargetCheck, TargetSnapshot, EVIDENCE_SYNTHETIC_KEYS,
};

/// One scripted focused target.
#[derive(Debug, Clone)]
pub struct FakeTarget {
    pub app: Option<String>,
    pub title: Option<String>,
    pub pid: Option<u32>,
}

impl FakeTarget {
    /// A target of another app (pid 4213).
    pub fn named(app: &str, title: &str) -> FakeTarget {
        FakeTarget {
            app: Some(app.to_string()),
            title: Some(title.to_string()),
            pid: Some(4213),
        }
    }

    /// A target owned by the test process itself.
    pub fn owned_by_this_process() -> FakeTarget {
        FakeTarget {
            app: Some("Starling".to_string()),
            title: Some("Starling notes".to_string()),
            pid: Some(std::process::id()),
        }
    }
}

/// What `insert` does once the guards pass.
#[derive(Debug, Clone)]
pub enum InsertBehavior {
    Type,
    FailWith(InsertError),
}

struct State {
    availability: Result<(), InsertError>,
    focus: Option<FakeTarget>,
    /// Minted per [`FakeBackend::focus`] call, so repeated captures of one
    /// focus yield the same ref.
    focus_id: u32,
    destroyed_ref: Option<String>,
    insert_behavior: InsertBehavior,
    revalidate_failure: Option<InsertError>,
    verifies_target: bool,
    key_hook: Option<KeyHook>,
    insertions: Vec<(String, String)>,
}

/// Runs before each character `insert_guarded` types, with its index.
type KeyHook = Arc<dyn Fn(usize) + Send + Sync>;

impl State {
    fn current_ref(&self) -> Option<String> {
        let target = self.focus.as_ref()?;
        Some(format_ref(
            BackendKind::Fake,
            self.focus_id,
            self.focus_id,
            target.pid,
        ))
    }
}

pub struct FakeBackend {
    state: Mutex<State>,
    excluded_pids: Vec<u32>,
}

impl Default for FakeBackend {
    fn default() -> Self {
        FakeBackend::new()
    }
}

impl FakeBackend {
    /// A ready backend with nothing focused, refusing only this process.
    pub fn new() -> FakeBackend {
        FakeBackend::with_excluded_pids(Vec::new())
    }

    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> FakeBackend {
        FakeBackend {
            state: Mutex::new(State {
                availability: Ok(()),
                focus: None,
                focus_id: 0,
                destroyed_ref: None,
                insert_behavior: InsertBehavior::Type,
                revalidate_failure: None,
                verifies_target: true,
                key_hook: None,
                insertions: Vec::new(),
            }),
            excluded_pids: merge_excluded_pids(excluded_pids),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("fake backend state")
    }

    /// Focus `target`; refs captured under an earlier focus now revalidate
    /// as `Changed`.
    pub fn focus(&self, target: FakeTarget) {
        let mut state = self.state();
        state.focus_id += 1;
        state.focus = Some(target);
        state.destroyed_ref = None;
    }

    /// Clear focus entirely.
    pub fn blur(&self) {
        self.state().focus = None;
    }

    /// Close the focused window: its ref revalidates as `Gone`.
    pub fn destroy_target(&self) {
        let mut state = self.state();
        state.destroyed_ref = state.current_ref();
        state.focus = None;
    }

    pub fn set_availability(&self, availability: Result<(), InsertError>) {
        self.state().availability = availability;
    }

    pub fn set_insert_behavior(&self, behavior: InsertBehavior) {
        self.state().insert_behavior = behavior;
    }

    /// Make every `revalidate` (including the one inside `insert`) fail
    /// with `error` until called again with `None`.
    pub fn fail_revalidate(&self, error: Option<InsertError>) {
        self.state().revalidate_failure = error;
    }

    /// Behave like a backend without target identity (Wayland): only
    /// [`InsertionBackend::verifies_target`] changes, so tests can drive
    /// the callers' gating.
    pub fn set_verifies_target(&self, verifies: bool) {
        self.state().verifies_target = verifies;
    }

    /// Run `hook` with each character's index before `insert_guarded`
    /// checks its `stop` for that character: lets a test change what the
    /// caller sees mid-typing.
    pub fn on_key(&self, hook: impl Fn(usize) + Send + Sync + 'static) {
        self.state().key_hook = Some(Arc::new(hook));
    }

    /// The `(target_ref, text)` pairs `insert` accepted, in order.
    pub fn insertions(&self) -> Vec<(String, String)> {
        self.state().insertions.clone()
    }
}

impl InsertionBackend for FakeBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Fake
    }

    fn verifies_target(&self) -> bool {
        self.state().verifies_target
    }

    fn availability(&self) -> Result<(), InsertError> {
        self.state().availability.clone()
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let state = self.state();
        let (Some(target), Some(target_ref)) = (state.focus.clone(), state.current_ref()) else {
            return Err(InsertError::Rejected {
                reason: "no window has input focus".to_string(),
            });
        };
        if target
            .pid
            .is_some_and(|pid| self.excluded_pids.contains(&pid))
        {
            return Err(InsertError::TargetIsStarling);
        }
        Ok(TargetSnapshot {
            backend: BackendKind::Fake,
            target_ref,
            app: target.app,
            title: target.title,
            pid: target.pid,
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        let state = self.state();
        if let Some(error) = &state.revalidate_failure {
            return Err(error.clone());
        }
        if state.destroyed_ref.as_ref() == Some(&target.target_ref) {
            return Ok(TargetCheck::Gone);
        }
        let actual = state
            .current_ref()
            .unwrap_or_else(|| "fake:none".to_string());
        Ok(if actual == target.target_ref {
            TargetCheck::Same
        } else {
            TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual,
            }
        })
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        self.insert_guarded(target, text, &|| None)
    }

    /// Checks `stop` before every character, like the Wayland backend;
    /// the text is recorded only when every character went out.
    fn insert_guarded(
        &self,
        target: &TargetSnapshot,
        text: &str,
        stop: &dyn Fn() -> Option<InsertError>,
    ) -> Result<InsertReceipt, InsertError> {
        insertion_guards(text, target.pid, &self.excluded_pids)?;
        let hook = self.state().key_hook.clone();
        let total_chars = text.chars().count();
        for index in 0..total_chars {
            if let Some(hook) = &hook {
                hook(index);
            }
            match stop() {
                None => {}
                Some(error) if index == 0 => return Err(error),
                Some(error) => {
                    return Err(InsertError::PartialDelivery {
                        delivered_chars: index,
                        total_chars,
                        cause: Box::new(error),
                    })
                }
            }
        }
        match self.revalidate(target)? {
            TargetCheck::Same => {}
            TargetCheck::Changed { expected, actual } => {
                return Err(InsertError::TargetChanged { expected, actual })
            }
            TargetCheck::Gone => return Err(InsertError::TargetGone),
        }
        let mut state = self.state();
        // The live pid decides ownership, like the real backends' checks.
        let live_pid = state.focus.as_ref().and_then(|focus| focus.pid);
        if live_pid.is_some_and(|pid| self.excluded_pids.contains(&pid)) {
            return Err(InsertError::TargetIsStarling);
        }
        match state.insert_behavior.clone() {
            InsertBehavior::Type => {
                state
                    .insertions
                    .push((target.target_ref.clone(), text.to_string()));
                Ok(InsertReceipt {
                    evidence: EVIDENCE_SYNTHETIC_KEYS,
                })
            }
            InsertBehavior::FailWith(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_is_stable_and_refuses_nothing_focused_or_starling() {
        let fake = FakeBackend::new();
        assert!(matches!(fake.capture(), Err(InsertError::Rejected { .. })));

        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().unwrap();
        assert_eq!(target.app.as_deref(), Some("Notes"));
        assert_eq!(target.pid, Some(4213));
        assert_eq!(fake.capture().unwrap().target_ref, target.target_ref);

        fake.focus(FakeTarget::owned_by_this_process());
        assert_eq!(fake.capture(), Err(InsertError::TargetIsStarling));
    }

    #[test]
    fn revalidate_tracks_focus_changes_blur_and_destruction() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let notes = fake.capture().unwrap();
        assert_eq!(fake.revalidate(&notes), Ok(TargetCheck::Same));

        fake.focus(FakeTarget::named("Browser", "A tab"));
        let browser = fake.capture().unwrap();
        assert_eq!(
            fake.revalidate(&notes),
            Ok(TargetCheck::Changed {
                expected: notes.target_ref.clone(),
                actual: browser.target_ref.clone(),
            })
        );

        fake.blur();
        assert!(matches!(
            fake.revalidate(&browser),
            Ok(TargetCheck::Changed { actual, .. }) if actual == "fake:none"
        ));

        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let notes_again = fake.capture().unwrap();
        fake.destroy_target();
        assert_eq!(fake.revalidate(&notes_again), Ok(TargetCheck::Gone));
    }

    #[test]
    fn insert_records_only_what_passes_the_guards() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().unwrap();
        fake.insert(&target, "Café, bitte.").unwrap();
        assert_eq!(
            fake.insert(&target, "no\nenter"),
            Err(InsertError::MultilineUnsupported)
        );
        fake.focus(FakeTarget::named("Browser", "A tab"));
        assert!(matches!(
            fake.insert(&target, "somewhere else"),
            Err(InsertError::TargetChanged { .. })
        ));
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref.clone(), "Café, bitte.".to_string())]
        );
    }

    #[test]
    fn excluded_pids_refuse_at_capture_and_at_insert() {
        // A host-configured exclusion: pid 555 is the app, not this process.
        let fake = FakeBackend::with_excluded_pids(vec![555]);
        fake.focus(FakeTarget {
            app: None,
            title: None,
            pid: Some(555),
        });
        assert_eq!(fake.capture(), Err(InsertError::TargetIsStarling));

        // A target captured elsewhere is refused once its pid is excluded.
        let other = FakeBackend::new();
        other.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = other.capture().unwrap();
        let strict = FakeBackend::with_excluded_pids(vec![4213]);
        assert_eq!(
            strict.insert(&target, "hello"),
            Err(InsertError::TargetIsStarling)
        );
        assert!(strict.insertions().is_empty());
    }
}
