import type { TranscriptionProtocol } from "@starling/dictation";

/**
 * Every settings field the dialog edits, as one immutable configuration
 * (B06): a draft while the dialog is open, the committed values once a save
 * succeeded. Nothing edits a single field of the live configuration in place
 * anymore — settings move from draft to committed only as a whole object,
 * so Cancel discards everything and a failed save activates nothing.
 */
export interface SettingsSnapshot {
  endpoint: string;
  protocol: TranscriptionProtocol;
  model: string;
  streamLive: boolean;
  expectedTerms: string;
  refineBaseUrl: string;
  refineModel: string;
  refineApiKey: string;
  refineInstruction: string;
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
  protocol: "starling:protocol",
  model: "starling:model",
  streaming: "starling:streaming",
  terms: "starling:terms",
  refineBaseUrl: "starling:refine:baseUrl",
  refineModel: "starling:refine:model",
  refineInstruction: "starling:refine:instruction",
  refineApiKey: "starling:refine:apiKey",
  refineApiKeyEnc: "starling:refine:apiKeyEnc",
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

/** Where the refinement API key lands: keychain ciphertext, plaintext, or nowhere. */
export interface RefinementKeyPlan {
  /** Value for starling:refine:apiKey, or null to remove the entry. */
  plaintext: string | null;

  /** Value for starling:refine:apiKeyEnc, or null to remove the entry. */
  ciphertext: string | null;
}

/**
 * Decide the refinement key's destination. An empty key clears both entries
 * (nothing to encrypt, no residue to keep); keychain ciphertext wins and the
 * plaintext copy is removed; without ciphertext the key falls back to
 * plaintext — a documented, deliberate fallback, never a silent one. The
 * plan itself writes nothing: the settings transaction owns every write.
 */
export function refinementKeyPlan(apiKey: string, ciphertext: string | null): RefinementKeyPlan {
  if (apiKey === "") return { plaintext: null, ciphertext: null };

  return ciphertext === null
    ? { plaintext: apiKey, ciphertext: null }
    : { plaintext: null, ciphertext };
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
 * holds a half-applied endpoint/key/protocol combination. The caller commits
 * React state only after this returns ok, so state and storage activate the
 * configuration together or not at all.
 */
export function persistSettings(
  settings: SettingsSnapshot,
  keyPlan: RefinementKeyPlan,
  storage: SettingsStorage,
): SettingsPersistResult {
  const writes: ReadonlyArray<readonly [string, string | null]> = [
    [KEYS.endpoint, settings.endpoint],
    [KEYS.protocol, settings.protocol],
    [KEYS.model, settings.model],
    [KEYS.streaming, settings.streamLive ? "1" : "0"],
    [KEYS.terms, settings.expectedTerms],
    [KEYS.refineBaseUrl, settings.refineBaseUrl],
    [KEYS.refineModel, settings.refineModel],
    [KEYS.refineInstruction, settings.refineInstruction],
    [KEYS.refineApiKey, keyPlan.plaintext],
    [KEYS.refineApiKeyEnc, keyPlan.ciphertext],
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
    protocol: storage.getItem(KEYS.protocol) === "openai" ? "openai" : "starling",
    model: storage.getItem(KEYS.model) ?? DEFAULT_MODEL,
    streamLive: storage.getItem(KEYS.streaming) !== "0",
    expectedTerms: storage.getItem(KEYS.terms) ?? "",
    refineBaseUrl: storage.getItem(KEYS.refineBaseUrl) ?? "",
    refineModel: storage.getItem(KEYS.refineModel) ?? "",
    refineApiKey: storage.getItem(KEYS.refineApiKey) ?? "",
    refineInstruction: storage.getItem(KEYS.refineInstruction) ?? "",
  };
}
