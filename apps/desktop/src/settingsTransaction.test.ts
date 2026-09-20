import { describe, expect, it } from "vite-plus/test";

import {
  EMPTY_ENDPOINT_REASON,
  normalizeSettings,
  persistSettings,
  readCommittedSettings,
  refinementKeyPlan,
  secureStorageStatus,
  type SecureKeyStorage,
  type SettingsSnapshot,
  type SettingsStorage,
} from "./settingsTransaction";

function draft(overrides: Partial<SettingsSnapshot> = {}): SettingsSnapshot {
  return {
    endpoint: "http://127.0.0.1:8181",
    protocol: "starling",
    model: "parakeet",
    streamLive: true,
    expectedTerms: "auth, Starling",
    refineBaseUrl: "http://127.0.0.1:11434/v1",
    refineModel: "llama3.1",
    refineApiKey: "",
    refineInstruction: "",
    refineKeyPlaintextOptIn: false,
    ...overrides,
  };
}

/**
 * In-memory localStorage stand-in. `failSetItem` names the one key whose
 * write throws, simulating a quota or security failure mid-transaction.
 */
class MemoryStorage implements SettingsStorage {
  private entries = new Map<string, string>();

  failSetItem?: string;

  constructor(entries: Record<string, string> = {}) {
    for (const [key, value] of Object.entries(entries)) this.entries.set(key, value);
  }

  getItem(key: string): string | null {
    return this.entries.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    if (key === this.failSetItem) throw new Error("storage write refused");

    this.entries.set(key, value);
  }

  removeItem(key: string): void {
    this.entries.delete(key);
  }
}

describe("normalizeSettings", () => {
  it("trims the endpoint and drops one trailing slash, and trims the refine fields", () => {
    const normalized = normalizeSettings(
      draft({
        endpoint: "  http://127.0.0.1:9000/ ",
        refineBaseUrl: "  http://127.0.0.1:11434/v1/ ",
        refineModel: "  llama3.1  ",
      }),
    );

    expect(normalized).toEqual({
      ok: true,
      settings: draft({
        endpoint: "http://127.0.0.1:9000",
        refineBaseUrl: "http://127.0.0.1:11434/v1/",
        refineModel: "llama3.1",
      }),
    });
  });

  it("keeps protocol, model, streaming, terms, key, and instruction exactly as drafted", () => {
    const changed = draft({
      protocol: "openai",
      model: " whisper-large ",
      streamLive: false,
      expectedTerms: "x, y",
      refineApiKey: "sk-test",
      refineInstruction: " tighten ",
    });

    expect(normalizeSettings(changed)).toEqual({ ok: true, settings: changed });
  });

  it("rejects a blank endpoint with a reason instead of an empty configuration", () => {
    const normalized = normalizeSettings(draft({ endpoint: "   " }));

    expect(normalized.ok).toBe(false);

    if (!normalized.ok) expect(normalized.reason).toBe(EMPTY_ENDPOINT_REASON);
  });
});

describe("refinementKeyPlan", () => {
  it("clears both key entries when the key is empty, whatever the store says", () => {
    expect(refinementKeyPlan({ apiKey: "", secure: null, plaintextOptIn: true })).toEqual({
      outcome: "cleared",
      plaintext: null,
      ciphertext: null,
    });
  });

  it("prefers keychain ciphertext and drops the plaintext copy", () => {
    expect(
      refinementKeyPlan({
        apiKey: "sk-test",
        secure: { kind: "encrypted", ciphertext: "cipher-text" },
        plaintextOptIn: false,
      }),
    ).toEqual({ outcome: "encrypted", plaintext: null, ciphertext: "cipher-text" });
  });

  it("never falls back to plaintext without an explicit opt-in (B10)", () => {
    const stores: ReadonlyArray<SecureKeyStorage | null> = [
      { kind: "unavailable" },
      { kind: "unprotected", backend: "basic_text" },
      { kind: "failed" },
      null,
    ];

    for (const secure of stores) {
      const plan = refinementKeyPlan({
        apiKey: "sk-test",
        secure,
        plaintextOptIn: false,
      });

      expect(plan.outcome).not.toBe("encrypted");

      if (plan.outcome !== "blocked") expect(plan.plaintext).not.toBe("sk-test");
    }
  });

  it("blocks saving a NEW key that cannot be stored securely without opt-in", () => {
    const plan = refinementKeyPlan({
      apiKey: "sk-new",
      secure: { kind: "failed" },
      plaintextOptIn: false,
      retained: { key: "sk-old", form: "encrypted" },
    });

    expect(plan.outcome).toBe("blocked");

    if (plan.outcome === "blocked") {
      expect(plan.message).toContain("not saved");
      expect(plan.message).toContain("nothing was changed");
      expect(plan.message).toContain("Store API key unencrypted");
    }
  });

  it("keeps an UNCHANGED key session-only instead of deleting or downgrading it", () => {
    const plan = refinementKeyPlan({
      apiKey: "sk-same",
      secure: { kind: "unavailable" },
      plaintextOptIn: false,
      retained: { key: "sk-same", form: "encrypted" },
    });

    expect(plan.outcome).toBe("session-only");

    if (plan.outcome === "session-only") {
      expect(plan.plaintext).toBeUndefined();
      expect(plan.ciphertext).toBeUndefined();
      expect(plan.status).toContain("previously encrypted copy is untouched");
    }
  });

  it("keeps a first-time key session-only too: nothing at rest, nothing mismatched", () => {
    const plan = refinementKeyPlan({
      apiKey: "sk-first",
      secure: null,
      plaintextOptIn: false,
    });

    expect(plan.outcome).toBe("session-only");

    if (plan.outcome === "session-only") {
      expect(plan.plaintext).toBeUndefined();
      expect(plan.ciphertext).toBeUndefined();
      expect(plan.status).toContain("session only");
      expect(plan.status).toContain("not written to disk");
    }
  });

  it("describes a retained legacy plaintext copy accurately", () => {
    const plan = refinementKeyPlan({
      apiKey: "sk-legacy",
      secure: { kind: "unprotected", backend: "basic_text" },
      plaintextOptIn: false,
      retained: { key: "sk-legacy", form: "plaintext" },
    });

    expect(plan.outcome).toBe("session-only");

    if (plan.outcome === "session-only") {
      expect(plan.status).toContain("already stored unencrypted");
      expect(plan.status).toContain("left untouched");
    }
  });

  it("persists plaintext only with the explicit opt-in, and replaces stale ciphertext", () => {
    const plan = refinementKeyPlan({
      apiKey: "sk-test",
      secure: { kind: "unavailable" },
      plaintextOptIn: true,
    });

    expect(plan).toEqual({ outcome: "plaintext", plaintext: "sk-test", ciphertext: null });
  });
});

describe("secureStorageStatus", () => {
  it("names each protection outcome for the user", () => {
    expect(secureStorageStatus({ kind: "encrypted", ciphertext: "x" })).toContain("keychain");

    expect(secureStorageStatus({ kind: "unprotected", backend: "basic_text" })).toContain(
      "basic_text",
    );

    expect(secureStorageStatus({ kind: "unavailable" })).toContain("No encrypted storage");

    expect(secureStorageStatus({ kind: "failed" })).toContain("refused");

    expect(secureStorageStatus(null)).toContain("No OS keychain");
  });
});

describe("persistSettings", () => {
  it("writes the whole configuration so readCommittedSettings round-trips it", () => {
    const storage = new MemoryStorage();
    const settings = draft({ refineApiKey: "sk-test", refineKeyPlaintextOptIn: true });

    const result = persistSettings(
      settings,
      refinementKeyPlan({ apiKey: "sk-test", secure: null, plaintextOptIn: true }),
      storage,
    );

    expect(result).toEqual({ ok: true });
    expect(readCommittedSettings(storage, "/api")).toEqual(
      expect.objectContaining({ ...settings, refineApiKey: "sk-test" }),
    );
  });

  it("replaces prior values and clears stale key entries", () => {
    const storage = new MemoryStorage({
      "starling:endpoint": "http://old:8181",
      "starling:model": "old-model",
      "starling:refine:apiKey": "old-plaintext-key",
    });

    persistSettings(
      draft({ endpoint: "http://new:8181", model: "new-model" }),
      refinementKeyPlan({
        apiKey: "sk-test",
        secure: { kind: "encrypted", ciphertext: "cipher-text" },
        plaintextOptIn: false,
      }),
      storage,
    );

    const committed = readCommittedSettings(storage, "/api");

    expect(committed.endpoint).toBe("http://new:8181");
    expect(committed.model).toBe("new-model");
    expect(storage.getItem("starling:refine:apiKey")).toBeNull();
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBe("cipher-text");
  });

  it("refuses a blocked plan instead of persisting around it", () => {
    const storage = new MemoryStorage({ "starling:endpoint": "http://old:8181" });

    const result = persistSettings(
      draft({ endpoint: "http://new:8181" }),
      refinementKeyPlan({
        apiKey: "sk-new",
        secure: { kind: "failed" },
        plaintextOptIn: false,
        retained: { key: "sk-old", form: "encrypted" },
      }),
      storage,
    );

    expect(result.ok).toBe(false);
    expect(storage.getItem("starling:endpoint")).toBe("http://old:8181");
  });

  it("writes neither key entry for a session-only plan, so the last encrypted key survives", () => {
    // B10: a transient keychain failure must not delete the stored ciphertext
    // nor activate a half-applied endpoint/key combination.
    const storage = new MemoryStorage({ "starling:refine:apiKeyEnc": "old-cipher" });

    const result = persistSettings(
      draft({ endpoint: "http://new:8181", refineApiKey: "sk-same" }),
      refinementKeyPlan({
        apiKey: "sk-same",
        secure: { kind: "failed" },
        plaintextOptIn: false,
        retained: { key: "sk-same", form: "encrypted" },
      }),
      storage,
    );

    expect(result).toEqual({ ok: true });
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBe("old-cipher");
    expect(storage.getItem("starling:refine:apiKey")).toBeNull();
    expect(storage.getItem("starling:endpoint")).toBe("http://new:8181");
  });

  it("keeps a legacy plaintext entry untouched for a session-only plan", () => {
    const storage = new MemoryStorage({ "starling:refine:apiKey": "old-plaintext-key" });

    persistSettings(
      draft({ refineApiKey: "old-plaintext-key" }),
      refinementKeyPlan({
        apiKey: "old-plaintext-key",
        secure: { kind: "unprotected", backend: "basic_text" },
        plaintextOptIn: false,
        retained: { key: "old-plaintext-key", form: "plaintext" },
      }),
      storage,
    );

    expect(storage.getItem("starling:refine:apiKey")).toBe("old-plaintext-key");
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBeNull();
  });

  it("clears both stored forms when the key is emptied", () => {
    const storage = new MemoryStorage({
      "starling:refine:apiKey": "old-plaintext-key",
      "starling:refine:apiKeyEnc": "old-cipher",
    });

    persistSettings(
      draft({ refineApiKey: "" }),
      refinementKeyPlan({ apiKey: "", secure: null, plaintextOptIn: false }),
      storage,
    );

    expect(storage.getItem("starling:refine:apiKey")).toBeNull();
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBeNull();
  });

  it("round-trips the plaintext opt-in choice", () => {
    const storage = new MemoryStorage();

    persistSettings(
      draft({ refineKeyPlaintextOptIn: true }),
      refinementKeyPlan({ apiKey: "", secure: null, plaintextOptIn: true }),
      storage,
    );

    expect(readCommittedSettings(storage, "/api").refineKeyPlaintextOptIn).toBe(true);
    expect(storage.getItem("starling:refine:keyPlaintextOptIn")).toBe("1");

    persistSettings(
      draft({ refineKeyPlaintextOptIn: false }),
      refinementKeyPlan({ apiKey: "", secure: null, plaintextOptIn: false }),
      storage,
    );

    expect(readCommittedSettings(storage, "/api").refineKeyPlaintextOptIn).toBe(false);
    expect(storage.getItem("starling:refine:keyPlaintextOptIn")).toBeNull();
  });

  it("rolls every key back to its prior value when a write fails mid-transaction", () => {
    const storage = new MemoryStorage({
      "starling:endpoint": "http://old:8181",
      "starling:protocol": "starling",
      "starling:model": "old-model",
      "starling:refine:apiKey": "old-plaintext-key",
    });

    storage.failSetItem = "starling:model";

    const result = persistSettings(
      draft({ endpoint: "http://new:8181", model: "new-model" }),
      refinementKeyPlan({ apiKey: "sk-test", secure: null, plaintextOptIn: true }),
      storage,
    );

    expect(result.ok).toBe(false);

    if (!result.ok) expect(result.message).toContain("storage write refused");

    // Nothing partially applied: the older configuration is intact and the
    // keys that had no prior value are still absent.
    expect(storage.getItem("starling:endpoint")).toBe("http://old:8181");
    expect(storage.getItem("starling:model")).toBe("old-model");
    expect(storage.getItem("starling:streaming")).toBeNull();
    expect(storage.getItem("starling:refine:baseUrl")).toBeNull();
    expect(storage.getItem("starling:refine:apiKey")).toBe("old-plaintext-key");
  });

  it("restores a key entry a later failure invalidated after it was removed", () => {
    const storage = new MemoryStorage({ "starling:refine:apiKey": "old-plaintext-key" });
    // The plaintext entry is removed before the ciphertext write fails.
    storage.failSetItem = "starling:refine:apiKeyEnc";

    const result = persistSettings(
      draft(),
      refinementKeyPlan({
        apiKey: "sk-test",
        secure: { kind: "encrypted", ciphertext: "cipher-text" },
        plaintextOptIn: false,
      }),
      storage,
    );

    expect(result.ok).toBe(false);
    expect(storage.getItem("starling:refine:apiKey")).toBe("old-plaintext-key");
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBeNull();
  });
});

describe("readCommittedSettings", () => {
  it("falls back to the defaults on empty storage", () => {
    const committed = readCommittedSettings(new MemoryStorage(), "/api");

    expect(committed).toEqual({
      endpoint: "/api",
      protocol: "starling",
      model: "parakeet",
      streamLive: true,
      expectedTerms: "",
      refineBaseUrl: "",
      refineModel: "",
      refineApiKey: "",
      refineInstruction: "",
      refineKeyPlaintextOptIn: false,
    });
  });

  it("decodes stored values, defaulting an unknown protocol to starling", () => {
    const committed = readCommittedSettings(
      new MemoryStorage({
        "starling:endpoint": "http://127.0.0.1:8181",
        "starling:protocol": "openai",
        "starling:model": "whisper-1",
        "starling:streaming": "0",
        "starling:terms": "auth",
        "starling:refine:baseUrl": "http://127.0.0.1:11434/v1",
        "starling:refine:model": "llama3.1",
        "starling:refine:apiKey": "sk-test",
        "starling:refine:instruction": "tighten",
        "starling:refine:keyPlaintextOptIn": "1",
      }),
      "/api",
    );

    expect(committed.protocol).toBe("openai");
    expect(committed.model).toBe("whisper-1");
    expect(committed.streamLive).toBe(false);
    expect(committed.expectedTerms).toBe("auth");
    expect(committed.refineBaseUrl).toBe("http://127.0.0.1:11434/v1");
    expect(committed.refineModel).toBe("llama3.1");
    expect(committed.refineApiKey).toBe("sk-test");
    expect(committed.refineInstruction).toBe("tighten");
    expect(committed.refineKeyPlaintextOptIn).toBe(true);

    expect(
      readCommittedSettings(new MemoryStorage({ "starling:protocol": "bogus" }), "/api").protocol,
    ).toBe("starling");
  });
});
