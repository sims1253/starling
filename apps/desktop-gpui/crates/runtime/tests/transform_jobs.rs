//! Transform jobs (#294) on the live runtime: `jobs.transform` admitted
//! like a submit, `Loading → Transforming` on its own worker, streamed
//! output as `jobs.progress`, and `jobs.transformed{result}` or
//! `jobs.failed{reason}` at the end. The live event stream replays green
//! through the oracle port. Also: cancel stops the processor and nothing
//! is reported, a local-only request never reaches a remote provider, a
//! retry supersedes the job it retries, and the production
//! `PipelineProcessor` runs the builtin step and refuses providers it
//! does not have.

use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_processing::contract::{
    FailureReason, Locality, ProviderKind, ProviderRef, RequestContext, ResultStatus, TransformKind,
    TransformRequest,
};
use starling_processing::pipeline::Registry;
use starling_runtime::bus::{EventMessage, EventSub};
use starling_runtime::machine::Rejection;
use starling_runtime::protocol::replay::MachineReplay;
use starling_runtime::protocol::{Command, Event, JobLimits, RejectReason};
use starling_runtime::provider::{FakeProcessor, FakeTransform, PipelineProcessor, TransformProcessor};
use starling_runtime::{Runtime, RuntimeClient, RuntimeConfig};

fn request(id: &str, locality: Locality) -> TransformRequest {
    TransformRequest {
        schema_version: 1,
        request_id: id.to_string(),
        retry_of: None,
        draft_id: "draft-1".to_string(),
        capture_id: "cap-1".to_string(),
        base_revision: 1,
        source_attempt_ids: vec!["att-1".to_string()],
        mode_id: "clean".to_string(),
        mode_version: 1,
        prompt_version: None,
        kinds: vec![TransformKind::Clean],
        language: Some("en".to_string()),
        input: "um so the meeting is at three".to_string(),
        instruction: None,
        style: None,
        context: RequestContext::default(),
        provider: ProviderRef {
            id: "local-s1".to_string(),
            kind: ProviderKind::S1,
            locality,
            route: if locality == Locality::Local {
                "local-authoring-s1".to_string()
            } else {
                "remote-authoring-api".to_string()
            },
            model: "s1-mini".to_string(),
            artifact_sha256: None,
        },
        local_only: locality == Locality::Local,
        deadline_ms: 5_000,
        max_output_chars: 1_000,
    }
}

fn start(processor: Arc<dyn TransformProcessor>, limits: JobLimits) -> (Runtime, RuntimeClient, EventSub) {
    let config = RuntimeConfig::default()
        .with_processor(processor)
        .with_jobs_limits(limits);
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    (runtime, client, events)
}

fn limits(max_queued: u32, max_concurrent: u32) -> JobLimits {
    JobLimits {
        max_queued,
        max_concurrent,
        per_route: vec![],
    }
}

/// Collects events on `corr` until one of `terminal` arrives.
fn until_terminal(events: &EventSub, corr: &str, terminal: &[&str]) -> Vec<EventMessage> {
    let mut seen = Vec::new();
    let start = Instant::now();
    loop {
        if let Ok(message) = events.recv_timeout(Duration::from_millis(20)) {
            if message.corr.as_deref() == Some(corr) {
                let done = terminal.contains(&message.event.type_name());
                seen.push(message);
                if done {
                    return seen;
                }
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "no terminal event for {corr}; saw {:?}",
            seen.iter().map(|m| m.event.type_name()).collect::<Vec<_>>()
        );
    }
}

fn wait_for(what: &str, predicate: impl Fn() -> bool) {
    let start = Instant::now();
    while !predicate() {
        assert!(start.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_transform_job_runs_through_transforming_and_replays_green() {
    let processor = FakeProcessor::new(vec![FakeTransform {
        deltas: vec!["So the ".into(), "meeting is at three.".into()],
        outcome: Ok("So the meeting is at three.".into()),
        work_ms: 60,
    }]);
    let (runtime, client, events) = start(processor.clone(), limits(4, 1));
    let command = Command::JobsTransform {
        request: Box::new(request("req-1", Locality::Local)),
    };
    let command_value = serde_json::json!({
        "v": 1, "id": "cmd_t1", "ts": "2026-09-24T10:00:00Z", "corr": "job-t1",
        "type": command.type_name(), "payload": command.payload_value(),
    });
    client.send(Some("job-t1"), command).expect("transform accepted");
    let seen = until_terminal(&events, "job-t1", &["jobs.transformed", "jobs.failed"]);

    let types: Vec<&str> = seen.iter().map(|m| m.event.type_name()).collect();
    assert_eq!(types.first(), Some(&"jobs.queued"));
    assert_eq!(types.last(), Some(&"jobs.transformed"));
    assert!(types.contains(&"jobs.progress"), "streamed output: {types:?}");
    let Event::JobsTransformed(result) = &seen.last().unwrap().event else {
        unreachable!()
    };
    assert_eq!(result.status, ResultStatus::Completed);
    assert_eq!(result.text.as_deref(), Some("So the meeting is at three."));
    assert_eq!(result.request_id, "req-1");
    assert!(result.timing.queued_ms >= 0.0);
    assert!(result.timing.processing_ms > 0.0);

    // The live stream, with the internal dispatch chain stepped over,
    // replays green through the oracle port.
    let mut replay = MachineReplay::new("jobs", None).unwrap();
    replay.feed(&command_value).unwrap();
    let mut advanced = false;
    for message in &seen {
        if message.event.type_name() != "jobs.queued" && !advanced {
            for state in ["Dispatched", "Loading", "Transforming"] {
                replay.feed(&serde_json::json!({ "$advance": state })).unwrap();
            }
            advanced = true;
        }
        replay
            .feed(&message.to_value())
            .unwrap_or_else(|violation| panic!("{}: {violation}", message.event.type_name()));
    }
    replay.finish().expect("transform stream replays green");
    assert_eq!(client.snapshot().jobs.state, "Completed");
    assert_eq!(processor.requests().len(), 1);
    runtime.shutdown();
}

#[test]
fn a_provider_failure_ends_the_job_with_its_typed_reason() {
    let processor = FakeProcessor::new(vec![FakeTransform::fails(FailureReason::RateLimited, true)]);
    let (runtime, client, events) = start(processor, limits(4, 1));
    client
        .send(Some("job-f"), Command::JobsTransform {
            request: Box::new(request("req-f", Locality::Local)),
        })
        .unwrap();
    let seen = until_terminal(&events, "job-f", &["jobs.transformed", "jobs.failed"]);
    match &seen.last().unwrap().event {
        Event::JobsFailed { reason, retryable } => {
            assert_eq!(reason, "rate_limited");
            assert!(*retryable);
        }
        other => panic!("expected jobs.failed, got {other:?}"),
    }
    runtime.shutdown();
}

#[test]
fn cancel_stops_the_processor_and_nothing_is_reported() {
    let processor = FakeProcessor::new(vec![FakeTransform {
        deltas: vec![],
        outcome: Ok("too late".into()),
        work_ms: 30_000,
    }]);
    let (runtime, client, events) = start(processor.clone(), limits(4, 1));
    client
        .send(Some("job-c"), Command::JobsTransform {
            request: Box::new(request("req-c", Locality::Local)),
        })
        .unwrap();
    wait_for("the processor to start", || !processor.requests().is_empty());
    client
        .send(None, Command::JobsCancel {
            job_id: "job-c".into(),
        })
        .expect("cancel accepted");
    wait_for("the processor to return", || processor.returned("req-c"));
    // Nothing but the admission outcome ever lands on the job's stream.
    std::thread::sleep(Duration::from_millis(100));
    let mut types = Vec::new();
    while let Ok(message) = events.recv_timeout(Duration::from_millis(20)) {
        if message.corr.as_deref() == Some("job-c") {
            types.push(message.event.type_name());
        }
    }
    assert_eq!(types, vec!["jobs.queued"]);
    assert_eq!(client.snapshot().jobs.state, "Cancelled");
    runtime.shutdown();
}

#[test]
fn a_local_only_request_naming_a_remote_provider_is_refused_before_admission() {
    let processor = FakeProcessor::new(vec![FakeTransform::returns("never")]);
    let (runtime, client, _events) = start(processor.clone(), limits(4, 1));
    let mut sneaky = request("req-r", Locality::Remote);
    sneaky.local_only = true;
    let refused = client.send(Some("job-r"), Command::JobsTransform {
        request: Box::new(sneaky),
    });
    assert!(matches!(refused, Err(Rejection::RemoteForbidden { .. })), "{refused:?}");
    std::thread::sleep(Duration::from_millis(50));
    assert!(processor.requests().is_empty(), "nothing reached the processor");
    runtime.shutdown();
}

#[test]
fn a_retry_supersedes_the_job_it_retries() {
    let processor = FakeProcessor::new(vec![
        FakeTransform {
            deltas: vec![],
            outcome: Ok("first".into()),
            work_ms: 30_000,
        },
        FakeTransform::returns("second"),
    ]);
    let (runtime, client, events) = start(processor.clone(), limits(4, 1));
    client
        .send(Some("job-1"), Command::JobsTransform {
            request: Box::new(request("req-1", Locality::Local)),
        })
        .unwrap();
    wait_for("the first job to start", || !processor.requests().is_empty());
    let mut retry = request("req-2", Locality::Local);
    retry.retry_of = Some("req-1".to_string());
    client
        .send(Some("job-2"), Command::JobsTransform {
            request: Box::new(retry),
        })
        .unwrap();
    wait_for("the superseded job to stop", || processor.returned("req-1"));
    let seen = until_terminal(&events, "job-2", &["jobs.transformed", "jobs.failed"]);
    let Event::JobsTransformed(result) = &seen.last().unwrap().event else {
        panic!("the retry completes: {:?}", seen.last().unwrap().event)
    };
    assert_eq!(result.text.as_deref(), Some("second"));
    runtime.shutdown();
}

#[test]
fn the_same_request_twice_is_a_duplicate_submission() {
    let processor = FakeProcessor::new(vec![FakeTransform {
        deltas: vec![],
        outcome: Ok("one".into()),
        work_ms: 2_000,
    }]);
    let (runtime, client, events) = start(processor.clone(), limits(4, 1));
    client
        .send(Some("job-a"), Command::JobsTransform {
            request: Box::new(request("req-same", Locality::Local)),
        })
        .unwrap();
    client
        .send(Some("job-b"), Command::JobsTransform {
            request: Box::new(request("req-same", Locality::Local)),
        })
        .unwrap();
    let seen = until_terminal(&events, "job-b", &["jobs.rejected", "jobs.queued"]);
    assert!(matches!(
        seen.last().unwrap().event,
        Event::JobsRejected {
            reason: RejectReason::DuplicateSubmission
        }
    ));
    runtime.shutdown();
}

#[test]
fn transform_jobs_share_the_bounded_queue() {
    let processor = FakeProcessor::new(vec![
        FakeTransform {
            deltas: vec![],
            outcome: Ok("slow".into()),
            work_ms: 2_000,
        },
        FakeTransform::returns("queued"),
    ]);
    let (runtime, client, events) = start(processor.clone(), limits(1, 1));
    client
        .send(Some("q-1"), Command::JobsTransform {
            request: Box::new(request("req-q1", Locality::Local)),
        })
        .unwrap();
    wait_for("the first job to run", || !processor.requests().is_empty());
    client
        .send(Some("q-2"), Command::JobsTransform {
            request: Box::new(request("req-q2", Locality::Local)),
        })
        .unwrap();
    client
        .send(Some("q-3"), Command::JobsTransform {
            request: Box::new(request("req-q3", Locality::Local)),
        })
        .unwrap();
    let seen = until_terminal(&events, "q-3", &["jobs.rejected", "jobs.queued"]);
    assert!(matches!(
        seen.last().unwrap().event,
        Event::JobsRejected {
            reason: RejectReason::QueueFull
        }
    ));
    runtime.shutdown();
}

#[test]
fn the_pipeline_processor_runs_builtin_and_refuses_unknown_providers() {
    let processor = Arc::new(PipelineProcessor::new(Registry::default()));
    let (runtime, client, events) = start(processor, limits(4, 1));
    let mut builtin = request("req-b", Locality::Local);
    builtin.kinds = vec![];
    builtin.input = "first new line second".to_string();
    builtin.provider = ProviderRef {
        id: "builtin".into(),
        kind: ProviderKind::Builtin,
        locality: Locality::Local,
        route: "local-builtin".into(),
        model: "spoken-commands.v1".into(),
        artifact_sha256: None,
    };
    client
        .send(Some("job-b"), Command::JobsTransform {
            request: Box::new(builtin),
        })
        .unwrap();
    let seen = until_terminal(&events, "job-b", &["jobs.transformed", "jobs.failed"]);
    let Event::JobsTransformed(result) = &seen.last().unwrap().event else {
        panic!("{:?}", seen.last().unwrap().event)
    };
    // The request's input already went through the deterministic step
    // when it was built; the builtin run passes it through.
    assert_eq!(result.text.as_deref(), Some("first new line second"));

    client
        .send(Some("job-u"), Command::JobsTransform {
            request: Box::new(request("req-u", Locality::Local)),
        })
        .unwrap();
    let seen = until_terminal(&events, "job-u", &["jobs.transformed", "jobs.failed"]);
    match &seen.last().unwrap().event {
        Event::JobsFailed { reason, retryable } => {
            assert_eq!(reason, "provider_unavailable");
            assert!(!retryable);
        }
        other => panic!("expected provider_unavailable, got {other:?}"),
    }
    runtime.shutdown();
}
