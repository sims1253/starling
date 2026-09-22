import { describe, expect, it } from "vite-plus/test";

import indexHtml from "../index.html?raw";
import { streamWebSocketUrl } from "@starling/dictation";

import {
  STATIC_CONNECT_SRC,
  connectSrcAllows,
  loopbackStreamToken,
  loopbackStreamTokens,
  parseConnectSrcTokens,
  staticConnectSrcTokens,
} from "./connectPolicy";

/** The page origin the preview serves under (vite.config.ts: 127.0.0.1:1420). */
const PREVIEW_ORIGIN = "http://127.0.0.1:1420";

function allows(url: string, tokens: ReadonlyArray<string> = staticConnectSrcTokens()): boolean {
  return connectSrcAllows(tokens, url, PREVIEW_ORIGIN);
}

/** One directive's value out of a serialized CSP, the way the meta tag ships it. */
function directiveValue(policy: string, name: string): string {
  for (const directive of policy.split(";")) {
    const trimmed = directive.trim();

    if (trimmed.startsWith(`${name} `) || trimmed === name)
      return trimmed.slice(name.length).trim();
  }

  throw new Error(`the policy has no ${name} directive`);
}

/** The exact <meta> CSP the built renderer ships, read from index.html itself. */
function shippedPolicy(): string {
  const html = indexHtml;
  const content = /http-equiv="Content-Security-Policy"[^>]*content="([^"]*)"/.exec(html)?.[1];

  if (content === undefined) throw new Error("index.html carries no CSP meta tag");

  return content;
}

describe("the shipped connect-src (B01)", () => {
  it("is exactly the static policy connectPolicy derives from", () => {
    expect(directiveValue(shippedPolicy(), "connect-src")).toBe(STATIC_CONNECT_SRC);
  });

  it("never widens beyond loopback WebSocket origins beside http(s)", () => {
    expect(staticConnectSrcTokens()).toEqual([
      "'self'",
      "http:",
      "https:",
      "ws://127.0.0.1:*",
      "ws://localhost:*",
      "wss://127.0.0.1:*",
      "wss://localhost:*",
    ]);
  });

  it("keeps every other shipped directive untouched", () => {
    const policy = shippedPolicy();

    for (const name of ["default-src", "script-src", "font-src", "img-src", "media-src"]) {
      // style-src keeps its 'unsafe-inline' and is asserted below.
      expect(directiveValue(policy, name)).not.toBe("");
    }

    expect(directiveValue(policy, "style-src")).toBe("'self' 'unsafe-inline'");
  });
});

describe("the default live-transcription connection", () => {
  it("is the loopback ws:// URL StarlingStream derives from the default endpoint", () => {
    // App.tsx: the packaged default endpoint is http://127.0.0.1:8181.
    expect(streamWebSocketUrl("http://127.0.0.1:8181")).toBe("ws://127.0.0.1:8181/stream");
  });

  it("is permitted by the shipped policy (the B01 fix)", () => {
    expect(allows("ws://127.0.0.1:8181/stream")).toBe(true);
  });

  it("was blocked by the pre-fix policy — http:/https: never cover ws://", () => {
    expect(
      connectSrcAllows(["'self'", "http:", "https:"], "ws://127.0.0.1:8181/stream", PREVIEW_ORIGIN),
    ).toBe(false);
  });

  it("permits every CSP-nameable loopback host spelling over ws and wss on any port", () => {
    for (const scheme of ["ws", "wss"]) {
      for (const host of ["127.0.0.1", "localhost"]) {
        for (const port of ["8181", "65535", "9"]) {
          expect(allows(`${scheme}://${host}:${port}/stream`)).toBe(true);
        }
      }
    }
  });

  it("refuses the bracketed IPv6 spelling — CSP's host-source grammar cannot express it", () => {
    // Chromium drops `ws://[::1]:*` sources as invalid (QA round 1), and its
    // matcher compares hosts exactly, so no remaining token covers a
    // bracketed IPv6 URL. IPv6-loopback servers are reached through the
    // `localhost` spelling (which the OS resolves to ::1) or the native
    // bridge — never through a bracketed IPv6 token.
    expect(allows("ws://[::1]:8181/stream")).toBe(false);
    expect(allows("wss://[::1]:8181/stream")).toBe(false);
  });
});

describe("the browser preview under the same static policy", () => {
  it("permits the page-origin /api WebSocket proxy through 'self'", () => {
    // vite.config.ts proxies /api/stream to the backend with ws:true, so the
    // preview's path-only endpoint resolves against the page it is served from.
    const original = globalThis.location;

    Object.defineProperty(globalThis, "location", {
      value: new URL(`${PREVIEW_ORIGIN}/`),
      configurable: true,
    });

    try {
      expect(streamWebSocketUrl("/api")).toBe("ws://127.0.0.1:1420/api/stream");
    } finally {
      Object.defineProperty(globalThis, "location", { value: original, configurable: true });
    }

    expect(connectSrcAllows(["'self'"], "ws://127.0.0.1:1420/api/stream", PREVIEW_ORIGIN)).toBe(
      true,
    );
    expect(allows("ws://127.0.0.1:1420/api/stream")).toBe(true);
  });

  it("still permits the batch, health, and refinement http(s) fetches", () => {
    expect(allows("http://127.0.0.1:8181/inference")).toBe(true);
    expect(allows("http://127.0.0.1:8181/health")).toBe(true);
    expect(allows("https://api.openai.com/v1/chat/completions")).toBe(true);
    expect(allows("http://192.168.1.10:8181/inference")).toBe(true);
  });
});

describe("what the static policy must refuse", () => {
  it("rejects arbitrary LAN and internet ws:// hosts", () => {
    expect(allows("ws://192.168.1.10:8181/stream")).toBe(false);
    expect(allows("ws://10.0.0.5:9000/stream")).toBe(false);
    expect(allows("ws://dictation.example.com:8181/stream")).toBe(false);
  });

  it("rejects external wss:// hosts", () => {
    expect(allows("wss://streams.example.com/stream")).toBe(false);
    expect(allows("wss://127.0.0.1.evil.com/stream")).toBe(false);
    expect(allows("ws://evillocalhost:8181/stream")).toBe(false);
  });

  it("keeps the http↔ws scheme pairing separate in both directions", () => {
    expect(connectSrcAllows(["http:"], "ws://127.0.0.1:8181/stream", PREVIEW_ORIGIN)).toBe(false);
    expect(connectSrcAllows(["ws:"] as const, "http://127.0.0.1:8181/stream", PREVIEW_ORIGIN)).toBe(
      false,
    );
  });

  it("never downgrades: an https token does not permit http", () => {
    expect(connectSrcAllows(["https:"], "http://127.0.0.1:8181/stream", PREVIEW_ORIGIN)).toBe(
      false,
    );
    expect(connectSrcAllows(["http:"], "https://127.0.0.1:8181/stream", PREVIEW_ORIGIN)).toBe(true);
  });
});

describe("loopbackStreamTokens", () => {
  it("derives the ws token for a configured loopback endpoint", () => {
    expect(loopbackStreamTokens(["http://127.0.0.1:8181"])).toEqual(["ws://127.0.0.1:*"]);
    expect(loopbackStreamTokens(["https://127.0.0.1:8181"])).toEqual(["wss://127.0.0.1:*"]);
  });

  it("deduplicates host spellings and sorts, keeping the scheme split", () => {
    expect(
      loopbackStreamTokens([
        "http://localhost:1234",
        "http://localhost:4567",
        "https://localhost:9000",
      ]),
    ).toEqual(["ws://localhost:*", "wss://localhost:*"]);
  });

  it("yields no token for IPv6-literal loopback — the grammar cannot spell it", () => {
    // A bracketed token would be dropped by the browser (QA round 1), so
    // ::1-configured endpoints ride the native bridge like non-loopback ones.
    expect(loopbackStreamTokens(["http://[::1]:8181"])).toEqual([]);
  });

  it("yields no token for non-loopback endpoints — those belong to the native bridge", () => {
    expect(
      loopbackStreamTokens(["http://192.168.1.10:8181", "https://streams.example.com"]),
    ).toEqual([]);
  });

  it("derives tokens that actually permit the URLs they stand for", () => {
    const endpoints = ["http://127.0.0.1:8181", "https://localhost:9443"];
    const derived = parseConnectSrcTokens(loopbackStreamTokens(endpoints).join(" "));

    for (const endpoint of endpoints) {
      expect(connectSrcAllows(derived, streamWebSocketUrl(endpoint), PREVIEW_ORIGIN)).toBe(true);
    }

    expect(connectSrcAllows(derived, "ws://127.0.0.1:8182/stream", PREVIEW_ORIGIN)).toBe(true);
    expect(connectSrcAllows(derived, "http://127.0.0.1:8181/inference", PREVIEW_ORIGIN)).toBe(
      false,
    );
  });

  it("reports null for URLs a static token cannot cover", () => {
    expect(loopbackStreamToken("ws://192.168.1.10:8181/stream")).toBeNull();
    expect(loopbackStreamToken("wss://streams.example.com/stream")).toBeNull();
    expect(loopbackStreamToken("ws://localhost:8181/stream")).toBe("ws://localhost:*");
  });
});

describe("host-source port matching", () => {
  it("matches any port only for the * wildcard", () => {
    expect(
      connectSrcAllows(["ws://127.0.0.1:*"], "ws://127.0.0.1:8181/stream", PREVIEW_ORIGIN),
    ).toBe(true);
    expect(
      connectSrcAllows(["ws://127.0.0.1:8181"], "ws://127.0.0.1:8182/stream", PREVIEW_ORIGIN),
    ).toBe(false);
  });

  it("treats a token without a port as the scheme's default port only", () => {
    expect(connectSrcAllows(["ws://127.0.0.1"], "ws://127.0.0.1/stream", PREVIEW_ORIGIN)).toBe(
      true,
    );
    expect(connectSrcAllows(["ws://127.0.0.1"], "ws://127.0.0.1:8181/stream", PREVIEW_ORIGIN)).toBe(
      false,
    );
    expect(connectSrcAllows(["wss://localhost"], "wss://localhost/stream", PREVIEW_ORIGIN)).toBe(
      true,
    );
  });

  it("requires the host to match exactly, not as a suffix", () => {
    expect(
      connectSrcAllows(["ws://localhost:*"], "ws://notlocalhost:8181/stream", PREVIEW_ORIGIN),
    ).toBe(false);
  });
});
