"""Executable oracle for the Starling mode-routing contract (E18/E19/E26).

Semantic half of the frozen v1 contract in ``packages/contracts/
mode-routing/`` (the JSON schemas are the structural half). Deliberately
stdlib-only so any implementation - the native runtime, an editor adapter or
a third-party client - can be differentially tested against it.

``validate_config`` / ``prefix_match`` / ``resolve`` / ``selection_decision``
are a line-faithful port of the review package's executable behavioral spec
(``starling-final-package/reference/router.py``). Do not change their
semantics: they ARE the contract. Everything below the "additive helpers"
divider wraps that port (fixture loading, decision-record construction per
decision.schema.json, conflict extraction, snapshot expiry) without
altering it. There are no behavioral deviations from the reference; only
paths/docstrings were adapted to this repository.

Frozen precedence (MODES_AND_CONTEXT.md "Deterministic resolution", as
encoded by ``resolve``):

1. runtime/privacy/secure-field restrictions are constraints on every path
   (``policy:secure-field`` / blocked);
2. a locked explicit manual mode wins; its text stays unparsed for other
   modes (request ``manual_mode`` + ``manual_locked``, locked by default);
3. in an alias-enabled session (take not locked, resolved profile
   ``allow_spoken_overrides``, ``session_allows_aliases``) a leading
   literal escape selects verbatim payload handling;
4. an unambiguous registered leading alias selects its mode - longest token
   count, then character length; ties across different modes are
   ``phrase_conflict`` / needs_resolution, never a guess;
5. otherwise the frozen project/site/app rule by scope specificity
   (project > site > app) then explicit priority; equal-rank rules targeting
   different profiles are ``rule_conflict`` / needs_resolution; equal-rank
   same-profile ties break by rule id;
6. otherwise the global default profile.

Prefix grammar: leading position after permitted whitespace, whole-token
boundaries, finite separator set (space or colon), case-insensitive.
Quoted or mid-sentence mentions never trigger. Text spans use Unicode
code-point offsets, explicitly not native UTF-16 ranges. The routed payload
is a view with a recorded removed prefix span, never a destructive
mutation; the raw recognition text (and audio) is always preserved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "mode-routing"
FIXTURE_DIR = CONTRACT / "fixtures"

# --------------------------------------------------------------------------- #
# Reference port: pure profile/routing behavior for the supplied conservative
# leading-phrase contract. No microphone, network, model calls, file
# expansion or execution.
# --------------------------------------------------------------------------- #


def validate_config(config: dict[str, Any]) -> None:
    profiles = config.get('profiles', [])
    ids = [p['id'] for p in profiles]
    if len(ids) != len(set(ids)) or config.get('default_profile') not in ids:
        raise ValueError('Unique profiles and a valid default are required')
    if 'verbatim' not in ids:
        raise ValueError('The literal escape requires a verbatim profile')
    if any(r['profile_id'] not in ids for r in config.get('rules', [])):
        raise ValueError('Rule references an unknown profile')
    rule_ids = [r['id'] for r in config.get('rules', [])]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError('Duplicate rule ID')
    for p in profiles:
        if p['local_only'] and any(str(p.get(k, '')).startswith('remote-')
                                   for k in ('asr_route', 'authoring_route')):
            raise ValueError('A local-only profile cannot use a remote route')
        if p['selection_required'] and p['selected_text'] == 'off':
            raise ValueError('Required selection cannot be disabled')
        if p['delivery'] == 'replace_selection' and p['selected_text'] != 'edit_target':
            raise ValueError('Replacement requires edit-target authority')


def prefix_match(text: str, phrase: str) -> re.Match[str] | None:
    # Match only a leading sequence, never quoted or mid-sentence text.
    pattern = r'^\s*' + r'\s+'.join(re.escape(x) for x in phrase.strip().split())
    pattern += r'(?=$|[\s:])(?:\s*:\s*|\s+)?'
    return re.match(pattern, text, flags=re.IGNORECASE)


def resolve(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    raw = request['raw_text']
    if not isinstance(raw, str):
        raise ValueError('raw_text must be a string')
    if request.get('secure_field', False):
        return dict(mode=None, source='policy:secure-field', payload=raw, status='blocked')
    profiles = {p['id']: p for p in config['profiles']}
    manual = request.get('manual_mode')
    if manual is not None and manual not in profiles:
        raise ValueError('Unknown manual mode')
    chosen = manual or config['default_profile']
    source = 'manual' if manual else 'default'
    if manual is None:
        matched = []
        for rule in config['rules']:
            criteria = {k: rule[k] for k in ('project_id', 'site', 'app_id') if k in rule}
            if criteria and all(request.get(k) == v for k, v in criteria.items()):
                specificity = 3 if 'project_id' in criteria else 2 if 'site' in criteria else 1
                matched.append(((specificity, rule['priority']), rule))
        if matched:
            best = max(x[0] for x in matched)
            winners = [r for rank, r in matched if rank == best]
            if len({r['profile_id'] for r in winners}) > 1:
                return dict(mode=None, source='rule_conflict', payload=raw, status='needs_resolution')
            winner = sorted(winners, key=lambda x: x['id'])[0]
            chosen, source = winner['profile_id'], 'rule:' + winner['id']
    payload = raw
    prefix_span = None
    locked = manual is not None and request.get('manual_locked', True)
    aliases_allowed = (not locked and profiles[chosen]['allow_spoken_overrides']
                       and request.get('session_allows_aliases', True))
    if aliases_allowed:
        escape = prefix_match(raw, 'literal')
        if escape:
            chosen, source = 'verbatim', 'escape:literal'
            payload, prefix_span = raw[escape.end():], [0, escape.end()]
        else:
            matches = []
            for profile in profiles.values():
                for alias in profile['aliases']:
                    match = prefix_match(raw, alias)
                    if match:
                        matches.append((len(alias.strip().split()), len(alias.strip()), alias, profile['id'], match.end()))
            if matches:
                rank = max((x[0], x[1]) for x in matches)
                winners = [x for x in matches if (x[0], x[1]) == rank]
                if len({x[3] for x in winners}) > 1:
                    return dict(mode=None, source='phrase_conflict', payload=raw, status='needs_resolution')
                winner = sorted(winners, key=lambda x: (x[2], x[3]))[0]
                chosen, source = winner[3], 'phrase:' + winner[2]
                payload, prefix_span = raw[winner[4]:], [0, winner[4]]
    profile = profiles[chosen]
    needs_selection = profile['selection_required'] and not (
        request.get('selection_available', False) and request.get('selection_granted', False))
    status = 'needs_input' if not payload.strip() or needs_selection else 'ready'
    return dict(mode=chosen, source=source, payload=payload, status=status,
                raw_text=raw, prefix_span_codepoints=prefix_span,
                delivery=profile['delivery'], selected_text_role=profile['selected_text'],
                local_only=profile['local_only'])


def selection_decision(snapshot: dict[str, Any], current: dict[str, Any],
                       role: str, conditional_apply: bool) -> str:
    """A target-lease oracle only. It does not implement an atomic OS edit."""
    if snapshot.get('secure_field') or current.get('secure_field'):
        return 'blocked'
    if role != 'edit_target' or not conditional_apply:
        return 'preview'
    fields = ('app', 'document', 'version', 'range', 'digest', 'offset_encoding')
    if any(snapshot.get(k) is None or current.get(k) is None for k in fields):
        return 'preview'
    r = snapshot['range']
    if not (isinstance(r, list) and len(r) == 2 and all(type(x) is int for x in r) and 0 <= r[0] <= r[1]):
        raise ValueError('Invalid selection range')
    if snapshot['offset_encoding'] not in ('utf16', 'utf8_bytes', 'unicode_codepoints'):
        return 'preview'
    return 'apply' if all(snapshot[k] == current[k] for k in fields) else 'conflict'


# --------------------------------------------------------------------------- #
# Additive helpers (not part of the reference port; they never change the
# frozen resolve/selection semantics above).
# --------------------------------------------------------------------------- #


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_profiles(name: str = "profiles.json") -> dict[str, Any]:
    """A profiles resolution document from the contract fixtures."""
    return load_json(FIXTURE_DIR / name)


def load_schema(name: str) -> dict[str, Any]:
    return load_json(CONTRACT / name)


def routing_cases() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(config fixture name, config, case) for every routing fixture case.

    routing.json is the byte-faithful port of the reference 25 cases and
    always uses the canonical profiles.json. routing-variants.json adds the
    reference test_behavior.py config-mutation cases (frozen as variant
    profile documents) plus the explicit frozen-precedence cases; a case may
    name its config with a "profiles" key, defaulting to profiles.json.
    """
    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    configs: dict[str, dict[str, Any]] = {}
    for filename in ("routing.json", "routing-variants.json"):
        for case in load_json(FIXTURE_DIR / filename):
            config_name = case.get("profiles", "profiles.json")
            if config_name not in configs:
                configs[config_name] = load_profiles(config_name)
            cases.append((config_name, configs[config_name], case))
    return cases


def selection_cases() -> list[dict[str, Any]]:
    return load_json(FIXTURE_DIR / "selection.json")


def snapshot_expired(snapshot: dict[str, Any], at: str) -> bool:
    """True when ``at`` is past the snapshot's expires_at bound (RFC 3339
    strings with the same format sort lexicographically). A take that
    outlives its snapshot must re-snapshot, never reuse the frozen target."""
    expires = snapshot.get("expires_at")
    return expires is not None and at > expires


def _rule_winners(config: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    """Equal-best-rank rules for the request (the rule_conflict candidates)."""
    matched = []
    for rule in config['rules']:
        criteria = {k: rule[k] for k in ('project_id', 'site', 'app_id') if k in rule}
        if criteria and all(request.get(k) == v for k, v in criteria.items()):
            specificity = 3 if 'project_id' in criteria else 2 if 'site' in criteria else 1
            matched.append(((specificity, rule['priority']), rule))
    if not matched:
        return []
    best = max(x[0] for x in matched)
    return sorted((r for rank, r in matched if rank == best), key=lambda x: x['id'])


def _phrase_winners(config: dict[str, Any], raw: str) -> list[tuple[Any, ...]]:
    """Equal-best-rank leading-alias matches (the phrase_conflict candidates)."""
    matches = []
    for profile in config['profiles']:
        for alias in profile['aliases']:
            match = prefix_match(raw, alias)
            if match:
                matches.append((len(alias.strip().split()), len(alias.strip()),
                                alias, profile['id'], match.end()))
    if not matches:
        return []
    rank = max((x[0], x[1]) for x in matches)
    return sorted((x for x in matches if (x[0], x[1]) == rank), key=lambda x: (x[2], x[3]))


def conflicts_for(config: dict[str, Any], request: dict[str, Any],
                  result: dict[str, Any]) -> list[dict[str, Any]]:
    """decision.schema.json conflicts entries for a needs_resolution result."""
    if result['status'] != 'needs_resolution':
        return []
    if result['source'] == 'rule_conflict':
        winners = _rule_winners(config, request)
        if len({r['profile_id'] for r in winners}) > 1:
            return [{'kind': 'rule',
                     'candidates': [{'mode_id': r['profile_id'], 'via': r['id']}
                                    for r in winners]}]
        return []
    if result['source'] == 'phrase_conflict':
        winners = _phrase_winners(config, request['raw_text'])
        if len({x[3] for x in winners}) > 1:
            return [{'kind': 'phrase',
                     'candidates': [{'mode_id': x[3], 'via': x[2]} for x in winners]}]
        return []
    return []


def _rule_kind(rule: dict[str, Any]) -> str:
    if 'project_id' in rule:
        return 'project_rule'
    if 'site' in rule:
        return 'site_rule'
    return 'app_rule'


def explain(result: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Deterministic, content-free route explanation for overlay/history.

    Names modes, rules and configured aliases only - never payload or
    selection contents (E26: no sensitive selection contents in logs).
    """
    source = result['source']
    mode = result['mode']
    if source == 'manual':
        return f'manual mode {mode} for this take; text not parsed for other modes'
    if source == 'default':
        return f'no rule or leading phrase matched; default mode {mode}'
    if source.startswith('rule:'):
        rule_id = source.split(':', 1)[1]
        rule = next((r for r in (config or {}).get('rules', []) if r['id'] == rule_id), None)
        kind = _rule_kind(rule).replace('_', ' ') if rule else 'rule'
        return f'{kind} {rule_id} selected mode {mode}'
    if source.startswith('phrase:'):
        alias = source.split(':', 1)[1]
        return f'leading phrase "{alias}" selected mode {mode}'
    if source == 'escape:literal':
        return 'leading literal escape; payload handled verbatim'
    if source == 'policy:secure-field':
        return 'secure field: routing suppressed by policy'
    if source == 'rule_conflict':
        return 'equal-rank rules conflict; resolution required'
    if source == 'phrase_conflict':
        return 'equal-rank leading phrases conflict; resolution required'
    return f'routed by {source}'


def decision_from_resolve(config: dict[str, Any], request: dict[str, Any],
                          result: dict[str, Any], *, decision_id: str,
                          capture_id: str, raw_attempt_id: str,
                          context_snapshot_id: str | None = None) -> dict[str, Any]:
    """Lift a resolve() result into a decision.schema.json v1 record.

    The decision is a view contract: it preserves the raw attempt by
    reference (raw_attempt_id), records the removed prefix span and cannot
    express any provider/permission change - the frozen route and consent
    live in the activation-time context snapshot and the runtime protocol.
    """
    source_map = {
        'manual': 'manual',
        'default': 'default',
        'escape:literal': 'escape',
        'policy:secure-field': 'policy',
        'rule_conflict': 'conflict',
        'phrase_conflict': 'conflict',
    }
    raw_source = result['source']
    if raw_source in source_map:
        source = source_map[raw_source]
    elif raw_source.startswith('rule:'):
        rule = next((r for r in config['rules'] if r['id'] == raw_source.split(':', 1)[1]), None)
        source = _rule_kind(rule) if rule is not None else 'app_rule'
    elif raw_source.startswith('phrase:'):
        source = 'phrase'
    else:  # pragma: no cover - resolve() only emits the sources above
        raise ValueError(f'unknown resolve source {raw_source!r}')
    mode_id = result['mode']
    mode_version = None
    if mode_id is not None:
        profile = next(p for p in config['profiles'] if p['id'] == mode_id)
        mode_version = profile.get('version', 1)
    span = result.get('prefix_span_codepoints')
    return {
        'schema_version': 1,
        'decision_id': decision_id,
        'capture_id': capture_id,
        'raw_attempt_id': raw_attempt_id,
        'mode_id': mode_id,
        'mode_version': mode_version,
        'source': source,
        'status': result['status'],
        'matched_prefix_span': list(span) if span is not None else None,
        'span_encoding': 'unicode_codepoints',
        'explanation': explain(result, config),
        'payload_view': {
            'raw_preserved': True,
            'text': result['payload'],
            'removed_prefix_span': list(span) if span is not None else None,
        },
        'conflicts': conflicts_for(config, request, result),
        'context_snapshot_id': context_snapshot_id,
    }


# --------------------------------------------------------------------------- #
# Processing (#293/#294): additive rules for the mode entry's processing
# fields and the provider choice. They never change resolve(): routing picks
# the mode, these pick at most one provider for its authoring route.
# --------------------------------------------------------------------------- #

INSTRUCTION_KINDS = ("rewrite", "translate")


def validate_processing(config: dict[str, Any]) -> None:
    """Cross-field rules for the processing fields a schema cannot express."""
    for p in config.get('profiles', []):
        kinds = p['transform_kinds']
        if len(set(kinds)) != len(kinds) or len(set(p['context_fields'])) != len(p['context_fields']):
            raise ValueError(f'{p["id"]}: duplicate transform kind or context field')
        if kinds and p['authoring_route'] is None:
            raise ValueError(f'{p["id"]}: processing needs an authoring route')
        if p['behavior'] == 'verbatim' and (kinds or p['spoken_commands']):
            raise ValueError(f'{p["id"]}: verbatim returns raw text untouched')
        if p['style'] is not None and not {'clean', 'format'} & set(kinds):
            raise ValueError(f'{p["id"]}: style applies to clean/format only')
        if p['delivery'] == 'insert_enter':
            # Enter is an explicit, per-mode opt-in: never the default,
            # never on an edit target.
            if p['id'] == config.get('default_profile'):
                raise ValueError('insert_enter cannot be the default delivery')
            if p['selected_text'] == 'edit_target':
                raise ValueError('insert_enter cannot replace a selection')


def validate_provider(provider: dict[str, Any]) -> None:
    remote_route = provider['route'].startswith('remote-')
    if remote_route != (provider['locality'] == 'remote'):
        raise ValueError(f'{provider["id"]}: locality must match the route')
    if set(INSTRUCTION_KINDS) & set(provider['transform_kinds']) and not provider['instructions']:
        raise ValueError(f'{provider["id"]}: rewrite/translate need instructions')
    if (provider['kind'] == 'builtin') != (not provider['transform_kinds']):
        raise ValueError(f'{provider["id"]}: only the builtin provider runs no model kinds')
    if provider['kind'] == 'builtin' and provider['locality'] != 'local':
        raise ValueError(f'{provider["id"]}: the builtin step is local')
    if provider['locality'] == 'remote' and provider['artifact'] is not None:
        raise ValueError(f'{provider["id"]}: a remote provider has no local artifact')


def _language_ok(language: str | None, accepted: list[str]) -> bool:
    if '*' in accepted:
        return True
    if language is None:
        return False
    primary = language.split('-', 1)[0]
    return language in accepted or primary in accepted


def processing_route(profile: dict[str, Any],
                     providers: list[dict[str, Any]]) -> dict[str, Any]:
    """The provider for this mode's processing, or why there is none.

    status ``none``: transcribe only. ``ready``: exactly this provider,
    with the context fields that will actually be sent (the mode's allowed
    fields the provider accepts). ``blocked``: nothing runs, the raw text
    stands; there is no fallback to any other provider.
    """
    def blocked(reason: str) -> dict[str, Any]:
        return {'status': 'blocked', 'provider': None, 'reason': reason, 'context_fields': []}

    kinds = profile['transform_kinds']
    if not kinds:
        return {'status': 'none', 'provider': None, 'reason': None, 'context_fields': []}
    candidates = [p for p in providers if p['route'] == profile['authoring_route']]
    if profile['local_only'] and (
            str(profile['authoring_route']).startswith('remote-')
            or any(p['locality'] == 'remote' for p in candidates)):
        return blocked('remote_forbidden')
    if not candidates:
        return blocked('provider_unavailable')
    if len(candidates) > 1:
        return blocked('provider_conflict')
    provider = candidates[0]
    if not set(kinds) <= set(provider['transform_kinds']):
        return blocked('unsupported_kind')
    if not _language_ok(profile['language'], provider['languages']):
        return blocked('unsupported_language')
    fields = [f for f in profile['context_fields'] if f in provider['context_fields']]
    return {'status': 'ready', 'provider': provider['id'], 'reason': None, 'context_fields': fields}


def check_request(request: dict[str, Any], profile: dict[str, Any],
                  provider: dict[str, Any]) -> list[str]:
    """Cross-field violations of a transform request against the mode it
    claims and the provider it names (empty when consistent)."""
    found = []
    if (request['mode_id'], request['mode_version']) != (profile['id'], profile['version']):
        found.append('request names a different mode version')
    if request['kinds'] != profile['transform_kinds']:
        found.append('request kinds differ from the mode')
    if request['language'] != profile['language']:
        found.append('request language differs from the mode')
    if request['local_only'] != profile['local_only']:
        found.append('request local_only differs from the mode')
    if request['local_only'] and request['provider']['locality'] != 'local':
        found.append('local_only request names a remote provider')
    ref = request['provider']
    for key in ('id', 'kind', 'locality', 'route', 'model'):
        if ref[key] != provider[key]:
            found.append(f'provider {key} differs from the declaration')
    expected_sha = provider['artifact']['sha256'] if provider['artifact'] else None
    if ref['artifact_sha256'] != expected_sha:
        found.append('provider artifact differs from the declaration')
    if provider['kind'] == 'builtin':
        # The deterministic step alone: no model kinds, nothing sent.
        if request['kinds'] or request['context']:
            found.append('the builtin provider runs no model and sends no context')
    else:
        route = processing_route(profile, [provider])
        if route['status'] != 'ready':
            found.append(f'mode cannot use this provider: {route["reason"]}')
        for field in request['context']:
            if field not in route['context_fields']:
                found.append(f'context field {field} was not allowed to be sent')
    if request['instruction'] is not None and not set(INSTRUCTION_KINDS) & set(request['kinds']):
        found.append('only rewrite/translate may carry an instruction')
    if request['style'] != profile['style']:
        found.append('request style differs from the mode')
    return found


def check_result(result: dict[str, Any]) -> list[str]:
    found = []
    if result['status'] == 'completed':
        if result['text'] is None or result['failure'] is not None:
            found.append('a completed result carries text and no failure')
    elif result['text'] is not None or result['failure'] is None:
        found.append('a failed or cancelled result carries a failure and no text')
    if (result['status'] == 'cancelled' and result['failure'] is not None
            and result['failure']['reason'] != 'cancelled'):
        found.append('a cancelled result has the cancelled reason')
    return found
