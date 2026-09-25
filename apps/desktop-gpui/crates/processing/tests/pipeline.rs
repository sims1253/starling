//! The pipeline end to end (#294): plan → request → provider → result →
//! draft, against the fake server. Cancel detaches the job, a late result
//! cannot insert, and raw text survives every failure. Also replays the
//! deterministic-step fixtures (`fixtures/spoken-commands.json`) the
//! Python oracle replays.

mod support;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use starling_processing::contract::{
    ContextField, FailureReason, Locality, ModeEntry, ProfilesDocument, ProviderDecl, ProviderKind,
    ResultStatus, RouteBlock, Snippet, TransformKind, TransformResult,
};
use starling_processing::pipeline::{
    self, build_request, plan, Clock, ContextValues, Plan, Registry, RequestOptions,
};
use starling_processing::providers::openai::OpenAiProvider;
use starling_processing::providers::{ChatConfig, Provider};
use starling_processing::staging::{CommandKind, Draft, Outcome, ProposalStatus, ResultKind};
use starling_processing::{transforms, CancelToken};
use support::*;

/// The contract lives in the repository, read in place so the port and the
/// Python oracle test the same files.
fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../packages/contracts/mode-routing")
}

fn fixture(name: &str) -> Value {
    let path = contract_dir().join("fixtures").join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("contract fixture {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("contract fixture {}: {error}", path.display()))
}

#[test]
fn spoken_command_fixtures_replay_like_the_oracle() {
    let cases = fixture("spoken-commands.json");
    let mut failures = Vec::new();
    for case in cases.as_array().unwrap() {
        let snippets: Vec<Snippet> = serde_json::from_value(case["snippets"].clone()).unwrap();
        let got = transforms::apply(
            case["input"].as_str().unwrap(),
            case["language"].as_str(),
            case["spoken_commands"].as_bool().unwrap(),
            &snippets,
        );
        if got != case["output"].as_str().unwrap() {
            failures.push(format!(
                "{}: got {got:?}, want {:?}",
                case["name"], case["output"]
            ));
        }
    }
    assert!(cases.as_array().unwrap().len() >= 25);
    assert!(failures.is_empty(), "{failures:#?}");
}

fn clean_mode(route: &str, local_only: bool) -> ModeEntry {
    let doc: ProfilesDocument = serde_json::from_value(fixture("profiles.json")).unwrap();
    let mut mode = doc.profile("capture").unwrap().clone();
    mode.id = "clean".to_string();
    mode.authoring_route = Some(route.to_string());
    mode.local_only = local_only;
    mode.language = None;
    mode.vocabulary = vec!["Starling".to_string()];
    mode.context_fields = vec![ContextField::Vocabulary];
    mode
}

fn chat_decl(locality: Locality) -> ProviderDecl {
    ProviderDecl {
        schema_version: 1,
        id: "chat".to_string(),
        route: if locality == Locality::Local {
            "local-authoring-chat".to_string()
        } else {
            "remote-authoring-chat".to_string()
        },
        kind: ProviderKind::OpenaiCompatible,
        locality,
        transform_kinds: vec![
            TransformKind::Clean,
            TransformKind::Format,
            TransformKind::Rewrite,
        ],
        languages: vec!["*".to_string()],
        instructions: true,
        context_fields: vec![ContextField::Vocabulary],
        model: "m".to_string(),
        artifact: None,
        max_input_chars: 10_000,
    }
}

fn chat(server: &FakeServer, locality: Locality) -> Arc<dyn Provider> {
    let mut config = ChatConfig::new(format!("{}/v1", server.url()));
    config.stream = false;
    Arc::new(OpenAiProvider::new(chat_decl(locality), config).unwrap())
}

fn options(id: &str) -> RequestOptions {
    RequestOptions {
        request_id: id.to_string(),
        retry_of: None,
        deadline_ms: 3_000,
        max_output_chars: 2_000,
    }
}

fn answer(text: &str) -> Step {
    ok_json(json!({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}))
}

fn raw_draft() -> Draft {
    let mut draft = Draft::new("draft-1", "cap-1");
    draft.partial(0, "um so");
    draft.final_attempt(0, "att-1", "um so the meeting is at three comma right");
    draft
}

/// Runs the plan's model step for the draft's current revision.
fn process(
    draft: &Draft,
    mode: &ModeEntry,
    registry: &Registry,
    id: &str,
    cancel: &CancelToken,
) -> TransformResult {
    let Plan::Model {
        provider,
        context_fields,
    } = plan(mode, registry)
    else {
        panic!("expected a model plan");
    };
    let request = build_request(
        draft,
        mode,
        provider.declaration(),
        &context_fields,
        &ContextValues::default(),
        &options(id),
    );
    pipeline::run(
        &request,
        Some(provider.as_ref()),
        Clock::default(),
        &mut |_| {},
        cancel,
    )
}

fn feed(draft: &mut Draft, result: &TransformResult) -> Outcome {
    let kind = match result.status {
        ResultStatus::Completed => ResultKind::Completed,
        ResultStatus::Failed => ResultKind::Failed,
        ResultStatus::Cancelled => ResultKind::Cancelled,
    };
    draft.result(&result.request_id, kind, result.text.as_deref())
}

#[test]
fn the_request_carries_the_deterministic_step_and_only_allowed_context() {
    let server = FakeServer::start(vec![answer("So the meeting is at three, right.")]);
    let registry = Registry::new(vec![chat(&server, Locality::Local)]);
    let mode = clean_mode("local-authoring-chat", true);
    let mut draft = raw_draft();
    assert_eq!(draft.request_transform("r1", None), Outcome::Pending);
    let result = process(&draft, &mode, &registry, "r1", &CancelToken::new());
    assert_eq!(result.status, ResultStatus::Completed, "{result:?}");
    let sent = server.requests()[0].json();
    let user = sent["messages"][1]["content"].as_str().unwrap();
    assert!(
        user.contains("um so the meeting is at three, right"),
        "spoken commands ran first: {user}"
    );
    let system = sent["messages"][0]["content"].as_str().unwrap();
    assert!(
        system.contains("Starling"),
        "the allowed vocabulary field was sent"
    );
    assert_eq!(feed(&mut draft, &result), Outcome::Current);
    assert_eq!(
        draft.text(),
        "um so the meeting is at three comma right",
        "a result alone changes nothing"
    );
    assert_eq!(draft.accept("r1", false), Outcome::Applied);
    assert_eq!(draft.text(), "So the meeting is at three, right.");
    assert_eq!(
        draft.raw_text(),
        "um so the meeting is at three comma right"
    );
}

#[test]
fn a_late_result_after_an_edit_cannot_insert() {
    let server = FakeServer::start(vec![answer("Processed.")]);
    let registry = Registry::new(vec![chat(&server, Locality::Local)]);
    let mode = clean_mode("local-authoring-chat", true);
    let mut draft = raw_draft();
    draft.request_transform("r1", None);
    let snapshot = draft.clone();
    // The user edits while the job runs.
    draft.insert(0, "Note: ");
    let result = process(&snapshot, &mode, &registry, "r1", &CancelToken::new());
    assert_eq!(feed(&mut draft, &result), Outcome::Stale);
    assert_eq!(draft.accept("r1", false), Outcome::StaleRejected);
    assert_eq!(draft.swap_check("r1", "t1"), Outcome::Keep);
    assert_eq!(
        draft.text(),
        "Note: um so the meeting is at three comma right"
    );
}

#[test]
fn cancel_detaches_the_job_and_raw_stands() {
    let closed = Arc::new(std::sync::Mutex::new(false));
    let server = FakeServer::start(vec![slow_sse_until_closed(
        format!(
            "data: {}",
            json!({"choices": [{"delta": {"content": "x"}}]})
        ),
        Duration::from_millis(30),
        300,
        Arc::clone(&closed),
    )]);
    let mut config = ChatConfig::new(format!("{}/v1", server.url()));
    config.stream = true;
    let registry = Registry::new(vec![Arc::new(
        OpenAiProvider::new(chat_decl(Locality::Local), config).unwrap(),
    )]);
    let mode = clean_mode("local-authoring-chat", true);
    let mut draft = raw_draft();
    draft.request_transform("r1", None);
    let cancel = CancelToken::new();
    {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            cancel.cancel();
        });
    }
    let result = process(&draft, &mode, &registry, "r1", &cancel);
    assert_eq!(result.status, ResultStatus::Cancelled);
    assert_eq!(draft.cancel("r1"), Outcome::Cancelled);
    assert_eq!(feed(&mut draft, &result), Outcome::Discarded);
    // Even a completed answer arriving afterwards lands nowhere.
    assert_eq!(
        draft.result("r1", ResultKind::Completed, Some("late")),
        Outcome::Discarded
    );
    assert_eq!(draft.text(), draft.raw_text());
}

#[test]
fn raw_survives_every_failure() {
    let steps: Vec<Step> = vec![
        reply("429 Too Many Requests", vec![], "{}".to_string()),
        reply("500 Internal Server Error", vec![], "{}".to_string()),
        reply("302 Found", vec![("location", "/x")], String::new()),
        reply("200 OK", vec![], "not json".to_string()),
        ok_json(json!({"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]})),
        ok_json(json!({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})),
        reply(
            "404 Not Found",
            vec![],
            json!({"error": {"message": "model m not found", "code": "model_not_found"}})
                .to_string(),
        ),
        stall(Duration::from_secs(2)),
    ];
    let expected = [
        FailureReason::RateLimited,
        FailureReason::HttpError,
        FailureReason::RedirectBlocked,
        FailureReason::MalformedResponse,
        FailureReason::TruncatedOutput,
        FailureReason::EmptyOutput,
        FailureReason::UnknownModel,
        FailureReason::Timeout,
    ];
    let server = FakeServer::start(steps);
    let registry = Registry::new(vec![chat(&server, Locality::Local)]);
    let mode = clean_mode("local-authoring-chat", true);
    let mut draft = raw_draft();
    let raw = draft.text();
    for (index, want) in expected.iter().enumerate() {
        let id = format!("r{index}");
        draft.request_transform(&id, None);
        let mut request_options = options(&id);
        request_options.deadline_ms = 400;
        let Plan::Model {
            provider,
            context_fields,
        } = plan(&mode, &registry)
        else {
            panic!()
        };
        let request = build_request(
            &draft,
            &mode,
            provider.declaration(),
            &context_fields,
            &ContextValues::default(),
            &request_options,
        );
        let result = pipeline::run(
            &request,
            Some(provider.as_ref()),
            Clock::default(),
            &mut |_| {},
            &CancelToken::new(),
        );
        assert_eq!(
            result.failure.as_ref().map(|f| f.reason),
            Some(*want),
            "{result:?}"
        );
        assert_eq!(feed(&mut draft, &result), Outcome::Failed);
        assert_eq!(draft.text(), raw);
        assert_eq!(draft.raw_text(), raw);
        assert_eq!(draft.revision(), 2);
    }
}

#[test]
fn local_only_modes_never_plan_a_remote_provider() {
    let server = FakeServer::start(vec![]);
    let registry = Registry::new(vec![chat(&server, Locality::Remote)]);
    let blocked = clean_mode("remote-authoring-chat", true);
    assert!(matches!(
        plan(&blocked, &registry),
        Plan::Blocked(RouteBlock::RemoteForbidden)
    ));
    let allowed = clean_mode("remote-authoring-chat", false);
    let destination = plan(&allowed, &registry)
        .destination()
        .expect("a destination");
    assert_eq!(destination.0.locality, Locality::Remote);
    assert_eq!(destination.1, vec![ContextField::Vocabulary]);
}

#[test]
fn a_missing_provider_blocks_instead_of_falling_back() {
    let server = FakeServer::start(vec![]);
    let registry = Registry::new(vec![chat(&server, Locality::Local)]);
    let mode = clean_mode("local-authoring-other", true);
    assert!(matches!(
        plan(&mode, &registry),
        Plan::Blocked(RouteBlock::ProviderUnavailable)
    ));
}

#[test]
fn transcribe_only_and_builtin_plans() {
    let registry = Registry::default();
    let doc: ProfilesDocument = serde_json::from_value(fixture("profiles.json")).unwrap();
    let verbatim = doc.profile("verbatim").unwrap();
    assert!(matches!(plan(verbatim, &registry), Plan::Nothing));
    let faithful = doc.profile("faithful").unwrap();
    assert!(matches!(plan(faithful, &registry), Plan::Builtin));
    let mut draft = Draft::new("d", "c");
    draft.final_attempt(0, "a", "first new paragraph second");
    let request = build_request(
        &draft,
        faithful,
        &pipeline::builtin_declaration(),
        &[],
        &ContextValues::default(),
        &options("b1"),
    );
    assert!(request.kinds.is_empty());
    assert!(request.prompt_version.is_none());
    let result = pipeline::run(
        &request,
        None,
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    assert_eq!(result.text.as_deref(), Some("first\n\nsecond"));
    assert_eq!(result.provider.kind, ProviderKind::Builtin);
}

#[test]
fn instructions_travel_only_for_rewrite() {
    let mut draft = Draft::new("d", "c");
    draft.final_attempt(0, "a", "send the report make it formal");
    draft.mark_command(15, 30, CommandKind::TrailingInstruction);
    let mut mode = clean_mode("local-authoring-chat", true);
    let decl = chat_decl(Locality::Local);
    let clean = build_request(
        &draft,
        &mode,
        &decl,
        &[],
        &ContextValues::default(),
        &options("c1"),
    );
    assert_eq!(clean.input, "send the report");
    assert_eq!(clean.instruction, None);
    mode.transform_kinds = vec![TransformKind::Rewrite];
    mode.style = None;
    let rewrite = build_request(
        &draft,
        &mode,
        &decl,
        &[],
        &ContextValues::default(),
        &options("c2"),
    );
    assert_eq!(rewrite.instruction.as_deref(), Some(" make it formal"));
    assert!(
        rewrite.context.vocabulary.is_none(),
        "no field the plan did not allow"
    );
}

#[test]
fn results_validate_as_the_contract_records() {
    let registry = Registry::default();
    let doc: ProfilesDocument = serde_json::from_value(fixture("profiles.json")).unwrap();
    let faithful = doc.profile("faithful").unwrap();
    let mut draft = Draft::new("d", "c");
    draft.final_attempt(0, "a", "hello comma world");
    let request = build_request(
        &draft,
        faithful,
        &pipeline::builtin_declaration(),
        &[],
        &ContextValues::default(),
        &options("v1"),
    );
    assert!(matches!(plan(faithful, &registry), Plan::Builtin));
    let result = pipeline::run(
        &request,
        None,
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    // The schema's required keys, straight from the contract file.
    for (schema, value) in [
        (
            "transform-request.schema.json",
            serde_json::to_value(&request).unwrap(),
        ),
        (
            "transform-result.schema.json",
            serde_json::to_value(&result).unwrap(),
        ),
    ] {
        let schema: Value =
            serde_json::from_str(&std::fs::read_to_string(contract_dir().join(schema)).unwrap())
                .unwrap();
        let mut want: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut got: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);
    }
    assert_eq!(draft.proposal_status("v1"), None);
    draft.request_transform("v1", None);
    feed(&mut draft, &result);
    assert_eq!(draft.proposal_status("v1"), Some(ProposalStatus::Current));
}

#[test]
fn processing_recorded_events_match_the_insight_contract() {
    use starling_processing::insight::{Arrival, ProcessingRecorded};
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(contract_dir().join("../insight-events/schema.json")).unwrap(),
    )
    .unwrap();
    let branch = schema["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["properties"]["type"]["const"] == "processing_recorded")
        .unwrap();
    let mut want: Vec<&str> = branch["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    want.sort_unstable();

    let mut draft = Draft::new("d", "cap:1");
    draft.final_attempt(0, "a", "hello");
    let doc: ProfilesDocument = serde_json::from_value(fixture("profiles.json")).unwrap();
    let mut request = build_request(
        &draft,
        doc.profile("faithful").unwrap(),
        &pipeline::builtin_declaration(),
        &[],
        &ContextValues::default(),
        &options("req:1"),
    );
    request.capture_id = "cap:1".into();
    let result = pipeline::run(
        &request,
        None,
        Clock::default(),
        &mut |_| {},
        &CancelToken::new(),
    );
    let event =
        ProcessingRecorded::new(&request, &result, Arrival::Current, "2026-09-24T10:00:00Z");
    let value = serde_json::to_value(&event).unwrap();
    let mut got: Vec<&str> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    got.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(event.job_id, "req_1");
    assert_eq!(event.capture_id, "cap_1");
    assert_eq!(event.output_chars, Some(5));

    // The contract's own fixture events parse into the typed record.
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            contract_dir().join("../insight-events/fixtures/processing-latency.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for item in fixture.as_array().unwrap() {
        if item["type"] == "processing_recorded" {
            let parsed: ProcessingRecorded = serde_json::from_value(item.clone()).unwrap();
            assert_eq!(&serde_json::to_value(parsed).unwrap(), item);
        }
    }
}
