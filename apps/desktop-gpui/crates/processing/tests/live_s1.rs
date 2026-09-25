//! Live check against a real S1-mini starling-serve (#295). Skipped unless
//! `STARLING_S1_ENDPOINT` names one, e.g.
//!
//! ```text
//! starling-serve --model s1 --gguf models/s1-mini-bf16-exact.gguf --port 8182
//! STARLING_S1_ENDPOINT=http://127.0.0.1:8182 cargo test -p starling-processing --test live_s1 -- --nocapture
//! ```
//!
//! Offline processing through the provider and the draft, the empty
//! answer on filler-only input, and cancellation mid-decode.

use std::time::{Duration, Instant};

use starling_processing::contract::{
    Formality, Locality, ProviderDecl, ProviderKind, RequestContext, ResultStatus, Structure,
    Style, StyleContext, TransformKind, TransformRequest,
};
use starling_processing::pipeline::{self, Clock};
use starling_processing::providers::s1::S1Provider;
use starling_processing::staging::{Draft, Outcome, ResultKind};
use starling_processing::CancelToken;

fn endpoint() -> Option<String> {
    std::env::var("STARLING_S1_ENDPOINT")
        .ok()
        .filter(|value| !value.is_empty())
}

fn decl() -> ProviderDecl {
    ProviderDecl {
        schema_version: 1,
        id: "local-s1".to_string(),
        route: "local-authoring-s1".to_string(),
        kind: ProviderKind::S1,
        locality: Locality::Local,
        transform_kinds: vec![TransformKind::Clean, TransformKind::Format],
        languages: vec!["en".to_string()],
        instructions: false,
        context_fields: vec![],
        model: "s1-mini".to_string(),
        artifact: None,
        max_input_chars: 1_000_000,
    }
}

fn request(id: &str, input: &str, base_revision: u64) -> TransformRequest {
    TransformRequest {
        schema_version: 1,
        request_id: id.to_string(),
        retry_of: None,
        draft_id: "live".to_string(),
        capture_id: "live".to_string(),
        base_revision,
        source_attempt_ids: vec!["att-live".to_string()],
        mode_id: "clean-local".to_string(),
        mode_version: 1,
        prompt_version: None,
        kinds: vec![TransformKind::Clean],
        language: Some("en".to_string()),
        input: input.to_string(),
        instruction: None,
        style: Some(Style {
            formality: Formality::SemiFormal,
            structure: Structure::Prose,
            context: StyleContext::General,
        }),
        context: RequestContext::default(),
        provider: decl().reference(),
        local_only: true,
        deadline_ms: 120_000,
        max_output_chars: 4_000,
    }
}

#[test]
fn s1_mini_cleans_a_take_offline_as_a_proposal() {
    let Some(endpoint) = endpoint() else {
        eprintln!("STARLING_S1_ENDPOINT not set; skipping the live S1-mini check");
        return;
    };
    let provider = S1Provider::new(decl(), &endpoint).expect("loopback endpoint");
    let raw = "um so the meeting is at uh three no wait four pm on thursday";
    let mut draft = Draft::new("live", "live");
    draft.final_attempt(0, "att-live", raw);
    assert_eq!(draft.request_transform("live-1", None), Outcome::Pending);
    let started = Instant::now();
    let result = pipeline::run(
        &request("live-1", raw, draft.revision()),
        Some(&provider),
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    eprintln!(
        "S1-mini: {:?} in {:?} ({:?})",
        result.text,
        started.elapsed(),
        result.timing
    );
    assert_eq!(result.status, ResultStatus::Completed, "{result:?}");
    let text = result.text.clone().unwrap();
    assert!(!text.is_empty());
    let has_filler = text
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .any(|word| word.eq_ignore_ascii_case("um"));
    assert!(!has_filler, "fillers removed: {text}");
    assert_eq!(
        draft.result("live-1", ResultKind::Completed, Some(&text)),
        Outcome::Current
    );
    assert_eq!(draft.text(), raw, "a result alone changes nothing");
    assert_eq!(draft.accept("live-1", false), Outcome::Applied);
    assert_eq!(draft.raw_text(), raw);

    let filler = pipeline::run(
        &request("live-2", "um uh hmm", draft.revision()),
        Some(&provider),
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    assert_eq!(filler.status, ResultStatus::Completed, "{filler:?}");
    assert_eq!(
        filler.text.as_deref(),
        Some(""),
        "empty on filler-only input is valid"
    );
}

#[test]
fn s1_mini_cancel_returns_promptly_mid_decode() {
    let Some(endpoint) = endpoint() else {
        eprintln!("STARLING_S1_ENDPOINT not set; skipping the live S1-mini cancel check");
        return;
    };
    let provider = S1Provider::new(decl(), &endpoint).expect("loopback endpoint");
    let long =
        "so um i was thinking that we should probably uh move the the launch to next week because \
                the the tests are not done yet and uh also marketing said they need more time ";
    let cancel = CancelToken::new();
    {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1500));
            cancel.cancel();
        });
    }
    let started = Instant::now();
    let result = pipeline::run(
        &request("live-cancel", &long.repeat(3), 1),
        Some(&provider),
        Clock::default(),
        &mut |_| {},
        &cancel,
    );
    let elapsed = started.elapsed();
    eprintln!("cancelled after {elapsed:?}: {:?}", result.failure);
    assert_eq!(result.status, ResultStatus::Cancelled, "{result:?}");
    assert!(elapsed < Duration::from_secs(4), "{elapsed:?}");
    // The server is free again: a short request right after succeeds.
    let after = pipeline::run(
        &request("live-after", "hello there", 1),
        Some(&provider),
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    assert_eq!(after.status, ResultStatus::Completed, "{after:?}");
}
