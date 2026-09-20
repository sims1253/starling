import { describe, expect, it } from "vite-plus/test";

import { storeRefinementKeySafe, type SafeStorageView } from "./keyProtection.js";

/**
 * Scripted safeStorage stand-in: answers the availability questions and
 * records whether encryption was attempted (B10 — an unprotected backend
 * must never be asked to "encrypt", because its ciphertext would only
 * costume the key as protected).
 */
class FakeSafeStorage implements SafeStorageView {
  encryptCalls = 0;

  constructor(
    private readonly options: {
      available?: boolean;
      backend?: string;
      availableThrows?: boolean;
      backendThrows?: boolean;
      encryptThrows?: boolean;
    } = {},
  ) {}

  isEncryptionAvailable(): boolean {
    if (this.options.availableThrows) throw new Error("not ready");

    return this.options.available ?? false;
  }

  getSelectedStorageBackend(): string {
    if (this.options.backendThrows) throw new Error("backend unavailable");

    return this.options.backend ?? "unknown";
  }

  encryptString(plaintext: string): Buffer {
    this.encryptCalls += 1;

    if (this.options.encryptThrows) throw new Error("keychain locked");

    return Buffer.from(`ciphertext:${plaintext}`, "utf8");
  }
}

describe("storeRefinementKeySafe", () => {
  it("encrypts with a healthy libsecret-backed Linux store", () => {
    const storage = new FakeSafeStorage({ available: true, backend: "gnome_libsecret" });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(result.protection).toBe("encrypted");
    expect(result.backend).toBe("gnome_libsecret");
    expect(result.ciphertext).toEqual(expect.any(String));
    expect(storage.encryptCalls).toBe(1);
  });

  it("never asks a basic_text backend to encrypt, however available it claims to be", () => {
    // The B10 trap: on Linux without a secret store, isEncryptionAvailable()
    // returns true while safeStorage guards with a hardcoded password.
    const storage = new FakeSafeStorage({ available: true, backend: "basic_text" });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(result.ciphertext).toBeNull();
    expect(result.protection).toBe("unprotected");
    expect(result.backend).toBe("basic_text");
    expect(storage.encryptCalls).toBe(0);
  });

  it("treats an unidentified Linux backend as unprotected, not as a keychain", () => {
    const storage = new FakeSafeStorage({ available: true, backend: undefined });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(result.ciphertext).toBeNull();
    expect(result.protection).toBe("unprotected");
    expect(storage.encryptCalls).toBe(0);

    const throwing = new FakeSafeStorage({ available: true, backendThrows: true });

    expect(storeRefinementKeySafe(throwing, "linux", "sk-test").protection).toBe("unprotected");
  });

  it("reports an unavailable store when the backend exists but the secret key does not", () => {
    // kwallet selected but the wallet is closed/locked: no secret key.
    const storage = new FakeSafeStorage({ available: false, backend: "kwallet5" });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(result).toEqual({ ciphertext: null, protection: "unavailable" });
    expect(storage.encryptCalls).toBe(0);
  });

  it("encrypts on non-Linux platforms when the OS store answers available", () => {
    const mac = new FakeSafeStorage({ available: true });

    expect(storeRefinementKeySafe(mac, "darwin", "sk-test")).toEqual({
      ciphertext: expect.any(String),
      protection: "encrypted",
    });

    const windows = new FakeSafeStorage({ available: true });

    expect(storeRefinementKeySafe(windows, "win32", "sk-test").protection).toBe("encrypted");
  });

  it("reports unavailable when the OS store is locked or the probe itself throws", () => {
    const locked = new FakeSafeStorage({ available: false });

    expect(storeRefinementKeySafe(locked, "darwin", "sk-test")).toEqual({
      ciphertext: null,
      protection: "unavailable",
    });

    const throwing = new FakeSafeStorage({ availableThrows: true });

    expect(storeRefinementKeySafe(throwing, "win32", "sk-test").protection).toBe("unavailable");
  });

  it("surfaces an encrypting store that throws as failed, not as ciphertext", () => {
    // The user denied access between the availability probe and the call.
    const storage = new FakeSafeStorage({
      available: true,
      backend: "gnome_libsecret",
      encryptThrows: true,
    });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(result).toEqual({ ciphertext: null, protection: "failed", backend: "gnome_libsecret" });
    expect(storage.encryptCalls).toBe(1);
  });

  it("base64-encodes the ciphertext so it survives the IPC boundary", () => {
    const storage = new FakeSafeStorage({ available: true, backend: "gnome_libsecret" });

    const result = storeRefinementKeySafe(storage, "linux", "sk-test");

    expect(Buffer.from(String(result.ciphertext), "base64").toString("utf8")).toBe(
      "ciphertext:sk-test",
    );
  });

  it("never includes the key itself in any reported field", () => {
    const unprotected = new FakeSafeStorage({ available: true, backend: "basic_text" });

    expect(
      JSON.stringify(storeRefinementKeySafe(unprotected, "linux", "sk-secret-value")),
    ).not.toContain("sk-secret-value");

    const failing = new FakeSafeStorage({ available: true, encryptThrows: true });

    expect(
      JSON.stringify(storeRefinementKeySafe(failing, "linux", "sk-secret-value")),
    ).not.toContain("sk-secret-value");
  });
});
