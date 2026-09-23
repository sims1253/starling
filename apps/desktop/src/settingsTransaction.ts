/**
 * Every settings field the dialog edits, as one immutable configuration
 * (B06): a draft while the dialog is open, the committed values once a save
 * succeeded. Nothing edits a single field of the live configuration in place
 * anymore — settings move from draft to committed only as a whole object,
 * so Cancel discards everything and a failed save activates nothing.
 */
export interface SettingsSnapshot {
  endpoint: string;
  model: string;
  streamLive: boolean;
  expectedTerms: string;
  refineBaseUrl: string;
  refineModel: string;
  refineApiKey: string;
  refineInstruction: string;
  /**
   * The explicit, informed choice to store the refinement API key as
   * plaintext when no OS secret store can encrypt it (B10). Defaults to
   * false: without it a key that cannot be encrypted is kept for the
   * session only and never written to disk.
   */
  refineKeyPlaintextOptIn: boolean;
}

/** Storage surface the settings transaction runs against. */
export interface SettingsStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** Every localStorage key a settings transaction owns (B06). */
const KEYS = {
  endpoint: "starling:endpoint",
  model: "starling:model",
  streaming: "starling:streaming",
  terms: "starling:terms",
  refineBaseUrl: "starling:refine:baseUrl",
  refineModel: "starling:refine:model",
  refineInstruction: "starling:refine:instruction",
  refineApiKey: "starling:refine:apiKey",
  refineApiKeyEnc: "starling:refine:apiKeyEnc",
  refineKeyPlaintextOptIn: "starling:refine:keyPlaintextOptIn",
} as const;

const DEFAULT_MODEL = "parakeet";

/** Why a draft cannot be saved at all: there is no endpoint to connect to. */
export const EMPTY_ENDPOINT_REASON = "Enter a server endpoint before saving.";

export type SettingsNormalization =
  | { ok: true; settings: SettingsSnapshot }
  | { ok: false; reason: string };

/**
 * Validate and normalize a whole draft at once (B06). The endpoint is
 * trimmed with one trailing slash dropped — the same shape the previous
 * save applied — and the refinement base URL and model are trimmed; every
 * other field is committed exactly as drafted. A draft without an endpoint
 * is refused outright instead of being half-activated.
 */
export function normalizeSettings(draft: SettingsSnapshot): SettingsNormalization {
  const endpoint = draft.endpoint.trim().replace(/\/$/, "");

  if (endpoint === "") return { ok: false, reason: EMPTY_ENDPOINT_REASON };

  return {
    ok: true,
    settings: {
      ...draft,
      endpoint,
      refineBaseUrl: draft.refineBaseUrl.trim(),
      refineModel: draft.refineModel.trim(),
    },
  };
}

/**
 * What the secure-save attempt concluded (B10): the desktop bridge's answer
 * after classifying the host's secret store, or null when there is no bridge
 * at all (browser preview) — a host with no keychain to ask.
 */
export type SecureKeyStorage =
  | { readonly kind: "encrypted"; readonly ciphertext: string }
  | { readonly kind: "unprotected" | "unavailable" | "failed"; readonly backend?: string };

/** The key already durably at rest, and in which form. */
export interface RetainedKey {
  readonly key: string;
  readonly form: "encrypted" | "plaintext";
}

/** The entries the plan wants written (string), removed (null), or left alone (undefined). */
interface KeyEntryPlan {
  /** Value for starling:refine:apiKey: set, remove, or leave untouched. */
  readonly plaintext: string | null | undefined;

  /** Value for starling:refine:apiKeyEnc: set, remove, or leave untouched. */
  readonly ciphertext: string | null | undefined;
}

export type RefinementKeyPlan = KeyEntryPlan &
  (
    | { readonly outcome: "cleared" }
    | { readonly outcome: "encrypted" }
    | { readonly outcome: "plaintext" }
    | { readonly outcome: "session-only"; readonly status: string }
    | { readonly outcome: "blocked"; readonly message: string }
  );

/** Human wording for each protection outcome, reused by every message below. */
export function secureStorageStatus(secure: SecureKeyStorage | null): string {
  if (secure === null) return "No OS keychain is available to this app.";

  switch (secure.kind) {
    case "encrypted":
      return "The key is stored encrypted via your OS keychain.";
    case "unprotected":
      return secure.backend === "basic_text"
        ? "This Linux desktop's safeStorage backend is basic_text, which guards with a hardcoded password instead of an OS secret store."
        : "The OS secret store could not be identified on this machine.";
    case "unavailable":
      return "No encrypted storage is available on this machine right now.";
    case "failed":
      return "The OS keychain refused to encrypt the key.";
  }
}

const SESSION_ONLY_FIRST_TIME =
  'Settings saved, but your API key was not: it is kept for this session only and was not written to disk, so it will be gone after a restart. Tick "Store API key unencrypted" to persist it, or re-enter it next time.';

/**
 * Decide the refinement key's destination (B10). An empty key clears both
 * entries; keychain ciphertext wins and migrates any plaintext copy away;
 * without usable encryption, plaintext is written ONLY on the explicit
 * opt-in — never as an automatic fallback. Otherwise the key stays
 * session-only: if it is unchanged from a copy already at rest, both stored
 * entries are left untouched (a transient keychain failure must not delete
 * the last valid encrypted key, and an unchanged key cannot mismatch the
 * saved endpoint); if it is a NEW key, the save is blocked — silently
 * swapping or dropping the stored key next to a freshly saved endpoint
 * would activate a mismatched endpoint/key pair.
 */
export function refinementKeyPlan(input: {
  readonly apiKey: string;
  readonly secure: SecureKeyStorage | null;
  readonly plaintextOptIn: boolean;
  readonly retained?: RetainedKey | undefined;
}): RefinementKeyPlan {
  const { apiKey, secure, plaintextOptIn, retained } = input;

  if (apiKey === "") return { outcome: "cleared", plaintext: null, ciphertext: null };

  if (secure?.kind === "encrypted")
    return { outcome: "encrypted", plaintext: null, ciphertext: secure.ciphertext };

  if (plaintextOptIn) return { outcome: "plaintext", plaintext: apiKey, ciphertext: null };

  const status = secureStorageStatus(secure);

  if (retained === undefined || retained.key === apiKey) {
    if (retained === undefined) {
      return {
        outcome: "session-only",
        plaintext: undefined,
        ciphertext: undefined,
        status: `${status} ${SESSION_ONLY_FIRST_TIME}`,
      };
    }

    const detail =
      retained.form === "encrypted"
        ? "The previously encrypted copy is untouched, so the key stays available after a restart."
        : 'Your API key was already stored unencrypted by an earlier version and was left untouched. Tick "Store API key unencrypted" to keep it deliberately, or clear the field to remove it.';

    return {
      outcome: "session-only",
      plaintext: undefined,
      ciphertext: undefined,
      status: `Settings saved. ${status} ${detail}`,
    };
  }

  return {
    outcome: "blocked",
    plaintext: undefined,
    ciphertext: undefined,
    message: `Your API key could not be stored securely — ${status.charAt(0).toLowerCase()}${status.slice(1)} The settings were not saved and nothing was changed. Unlock the keychain and try again, tick "Store API key unencrypted" to persist it as plaintext, or clear the key field.`,
  };
}

export type SettingsPersistResult = { ok: true } | { ok: false; message: string };

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * Persist a whole settings configuration as one all-or-nothing transition
 * (B06). The prior value of every affected key — including both refinement
 * key entries — is snapshotted first; if any write fails mid-way (quota,
 * private mode, disabled storage), the snapshot is restored so storage never
 * holds a half-applied endpoint/key combination. The caller commits
 * React state only after this returns ok, so state and storage activate the
 * configuration together or not at all.
 */
export function persistSettings(
  settings: SettingsSnapshot,
  keyPlan: RefinementKeyPlan,
  storage: SettingsStorage,
): SettingsPersistResult {
  if (keyPlan.outcome === "blocked") {
    // Defense in depth: the caller surfaces the message before ever starting
    // a transaction; if one arrives here anyway, nothing may be written
    // around a key decision that was never made (B10).
    return { ok: false, message: keyPlan.message };
  }

  // A session-only plan leaves both key entries untouched (undefined skips
  // the write), so a failed secure save cannot delete the last valid
  // encrypted key or strand a half-applied key form (B10).
  const writes: ReadonlyArray<readonly [string, string | null]> = [
    [KEYS.endpoint, settings.endpoint],
    [KEYS.model, settings.model],
    [KEYS.streaming, settings.streamLive ? "1" : "0"],
    [KEYS.terms, settings.expectedTerms],
    [KEYS.refineBaseUrl, settings.refineBaseUrl],
    [KEYS.refineModel, settings.refineModel],
    [KEYS.refineInstruction, settings.refineInstruction],
    [KEYS.refineKeyPlaintextOptIn, settings.refineKeyPlaintextOptIn ? "1" : null],
    ...(keyPlan.plaintext === undefined ? [] : ([[KEYS.refineApiKey, keyPlan.plaintext]] as const)),
    ...(keyPlan.ciphertext === undefined
      ? []
      : ([[KEYS.refineApiKeyEnc, keyPlan.ciphertext]] as const)),
  ];

  let before: ReadonlyArray<readonly [string, string | null]>;

  try {
    before = writes.map(([key]) => [key, storage.getItem(key)] as const);
  } catch (cause) {
    return { ok: false, message: messageFrom(cause) };
  }

  const write = (key: string, value: string | null) =>
    value === null ? storage.removeItem(key) : storage.setItem(key, value);

  try {
    for (const [key, value] of writes) write(key, value);

    return { ok: true };
  } catch (cause) {
    // Roll back to the snapshot: keys that held older values get them back,
    // keys that were absent are removed again. Restoration is best-effort —
    // storage that fails a write often fails the restore too — and the
    // returned message still names the original failure.
    for (const [key, prior] of before) {
      try {
        write(key, prior);
      } catch {
        /* best-effort rollback */
      }
    }

    return { ok: false, message: messageFrom(cause) };
  }
}

/**
 * Read the committed configuration (B06): one decoder for every settings
 * key, so a draft opened from it and a transaction written beside it can
 * never disagree about encoding. `defaultEndpoint` is environment-dependent
 * (desktop bridge or browser preview) and stays the caller's choice.
 */
export function readCommittedSettings(
  storage: SettingsStorage,
  defaultEndpoint: string,
): SettingsSnapshot {
  return {
    endpoint: storage.getItem(KEYS.endpoint) ?? defaultEndpoint,
    model: storage.getItem(KEYS.model) ?? DEFAULT_MODEL,
    streamLive: storage.getItem(KEYS.streaming) !== "0",
    expectedTerms: storage.getItem(KEYS.terms) ?? "",
    refineBaseUrl: storage.getItem(KEYS.refineBaseUrl) ?? "",
    refineModel: storage.getItem(KEYS.refineModel) ?? "",
    refineApiKey: storage.getItem(KEYS.refineApiKey) ?? "",
    refineInstruction: storage.getItem(KEYS.refineInstruction) ?? "",
    refineKeyPlaintextOptIn: storage.getItem(KEYS.refineKeyPlaintextOptIn) === "1",
  };
}
