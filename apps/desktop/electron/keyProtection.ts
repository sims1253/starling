import type { RefinementKeySaveResult } from "./ipc.js";

/**
 * Electron's safeStorage as this module consumes it (B10): an injectable
 * view so the classification and encryption decisions stay pure and
 * testable against scripted stores — healthy, locked, unavailable, throwing,
 * and the Linux basic_text trap.
 */
export interface SafeStorageView {
  isEncryptionAvailable(): boolean;
  getSelectedStorageBackend?(): string;
  encryptString(plaintext: string): Buffer;
}

/** What the host's safeStorage actually offers, beyond "can call encrypt". */
export type SecretStore =
  | { readonly protection: "keychain"; readonly backend?: string }
  | { readonly protection: "unavailable" }
  | {
      readonly protection: "unprotected";
      readonly reason: "basic-text" | "indeterminate";
      readonly backend: string;
    };

/**
 * Classify the host's secret store (B10). `isEncryptionAvailable()` alone is
 * not proof of protection: on Linux it returns true even when safeStorage
 * fell back to basic_text, which guards with a hardcoded password instead of
 * an OS secret store. So on Linux the selected backend decides first —
 * basic_text and an unidentified backend are unprotected — and only a real
 * secret-store backend (gnome_libsecret, kwallet…) proceeds to the
 * availability check. Elsewhere (macOS Keychain, Windows DPAPI) availability
 * is the whole question. Probes that throw read as their conservative
 * answer, never as a crash.
 */
export function classifySecretStore(
  platform: NodeJS.Platform,
  storage: SafeStorageView,
): SecretStore {
  let available = false;

  try {
    available = storage.isEncryptionAvailable();
  } catch {
    available = false;
  }

  if (platform !== "linux")
    return available ? { protection: "keychain" } : { protection: "unavailable" };

  let backend: string | undefined;

  try {
    backend = storage.getSelectedStorageBackend?.();
  } catch {
    backend = undefined;
  }

  if (backend === "basic_text") return { protection: "unprotected", reason: "basic-text", backend };

  // "unknown" (or a backend that could not be read) means Electron could not
  // identify a secret store; claiming keychain protection would be a guess.
  if (backend === undefined || backend === "unknown")
    return { protection: "unprotected", reason: "indeterminate", backend: backend ?? "unknown" };

  return available ? { protection: "keychain", backend } : { protection: "unavailable" };
}

/**
 * Store one refinement API key under the host's secret store (B10). Returns
 * ciphertext only when a real OS secret store encrypted it; every other
 * outcome reports its protection status so the renderer can require an
 * explicit choice before any plaintext persists. An unprotected backend is
 * never asked to encrypt — its ciphertext would only costume the key as
 * protected. No field ever carries the key itself.
 */
export function storeRefinementKeySafe(
  storage: SafeStorageView,
  platform: NodeJS.Platform,
  apiKey: string,
): RefinementKeySaveResult {
  const store = classifySecretStore(platform, storage);

  if (store.protection === "unprotected")
    return { ciphertext: null, protection: store.protection, backend: store.backend };

  if (store.protection === "unavailable") return { ciphertext: null, protection: store.protection };

  try {
    return {
      ciphertext: storage.encryptString(apiKey).toString("base64"),
      protection: "encrypted",
      backend: store.backend,
    };
  } catch {
    // The store answered available but refused the call — a locked keychain
    // or a denied access prompt. Honest failure, no ciphertext.
    return { ciphertext: null, protection: "failed", backend: store.backend };
  }
}
