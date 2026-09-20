import { describe, expect, it } from "vite-plus/test";

import {
  EMPTY_ENDPOINT_REASON,
  normalizeSettings,
  persistSettings,
  readCommittedSettings,
  refinementKeyPlan,
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
  it("clears both key entries when the key is empty", () => {
    expect(refinementKeyPlan("", "cipher-text")).toEqual({ plaintext: null, ciphertext: null });
  });

  it("prefers keychain ciphertext and drops the plaintext copy", () => {
    expect(refinementKeyPlan("sk-test", "cipher-text")).toEqual({
      plaintext: null,
      ciphertext: "cipher-text",
    });
  });

  it("falls back to plaintext when no ciphertext could be produced", () => {
    expect(refinementKeyPlan("sk-test", null)).toEqual({ plaintext: "sk-test", ciphertext: null });
  });
});

describe("persistSettings", () => {
  it("writes the whole configuration so readCommittedSettings round-trips it", () => {
    const storage = new MemoryStorage();
    const settings = draft({ refineApiKey: "sk-test" });
    const result = persistSettings(settings, refinementKeyPlan("sk-test", null), storage);

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
      refinementKeyPlan("sk-test", "cipher-text"),
      storage,
    );

    const committed = readCommittedSettings(storage, "/api");

    expect(committed.endpoint).toBe("http://new:8181");
    expect(committed.model).toBe("new-model");
    expect(storage.getItem("starling:refine:apiKey")).toBeNull();
    expect(storage.getItem("starling:refine:apiKeyEnc")).toBe("cipher-text");
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
      refinementKeyPlan("sk-test", null),
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

    const result = persistSettings(draft(), refinementKeyPlan("sk-test", "cipher-text"), storage);

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

    expect(
      readCommittedSettings(new MemoryStorage({ "starling:protocol": "bogus" }), "/api").protocol,
    ).toBe("starling");
  });
});
