import { streamWebSocketUrl } from "@starling/dictation";

/**
 * The renderer's connection policy for the WebSocket transport (B01).
 *
 * What the app connects to, derived from the sources that build the URLs:
 * - Live dictation opens `StarlingStream`, whose URL is derived by
 *   `streamWebSocketUrl` (packages/dictation/src/streaming.ts): an
 *   `http(s)://host[:port]` endpoint swaps its scheme to `ws(s)://`, and a
 *   path-only endpoint (the browser preview's `/api`) resolves against the
 *   page origin. The desktop default endpoint is `http://127.0.0.1:8181`
 *   (App.tsx), so the default live URL is `ws://127.0.0.1:8181/stream`.
 * - Batch transcription and health checks fetch `${endpoint}/inference` etc.
 *   directly from the renderer only in the browser preview (the packaged app
 *   routes them through the desktop bridge), and refinement always fetches an
 *   arbitrary user-configured OpenAI-compatible `http(s)` endpoint.
 *
 * A static CSP cannot enumerate user-configured LAN `ws://` or `wss://`
 * endpoints, so the policy is split: this static string permits only loopback
 * WebSocket origins (any port) beside the http(s) schemes the batch and
 * refinement paths already require, and the packaged app carries non-loopback
 * streaming over the narrow native transport bridge in
 * electron/streamBridge.ts, where the main process — not the renderer's CSP —
 * validates the endpoint. The browser preview's documented surface for
 * non-loopback streaming is the `/api` WebSocket dev proxy (vite.config.ts),
 * which is same-origin.
 */
export const STATIC_CONNECT_SRC =
  "'self' http: https: ws://127.0.0.1:* ws://localhost:* ws://[::1]:* wss://127.0.0.1:* wss://localhost:* wss://[::1]:*";

/** Hosts a static token may name: loopback only, so the ws surface stays narrow. */
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

/** Default ports per URL scheme, used to compare ports that were omitted. */
const DEFAULT_PORTS: Readonly<Record<string, string>> = {
  ftp: "21",
  http: "80",
  https: "443",
  ws: "80",
  wss: "443",
};

/** The shipped policy as a token list, the form the matcher consumes. */
export function staticConnectSrcTokens(): string[] {
  return STATIC_CONNECT_SRC.split(/\s+/);
}

/**
 * The CSP `host-source` token for one streaming URL, or null when the URL is
 * not loopback — non-loopback streaming is deliberately left to the packaged
 * app's native bridge instead of this static policy.
 */
export function loopbackStreamToken(streamUrl: string): string | null {
  const url = new URL(streamUrl);
  // URL.hostname keeps IPv6 brackets ("[::1]"); the loopback set does not.
  const host = unwrapIpv6(url.hostname);

  if (!LOOPBACK_HOSTS.has(host)) return null;

  return `${url.protocol}//${wrapIpv6(host)}:*`;
}

/**
 * Derive the loopback tokens for a set of configured endpoints: each endpoint
 * maps to the exact URL `StarlingStream` would open (`streamWebSocketUrl` is
 * the derivation the transport itself uses), and only its loopback forms yield
 * a static token. Path-only endpoints (the preview `/api`) resolve against the
 * page and are covered by `'self'`, not by a token.
 */
export function loopbackStreamTokens(endpoints: ReadonlyArray<string>): string[] {
  const tokens = new Set<string>();

  for (const endpoint of endpoints) {
    const token = loopbackStreamToken(streamWebSocketUrl(endpoint));

    if (token) tokens.add(token);
  }

  return [...tokens].sort();
}

/**
 * Parse the token grammar this policy uses out of a `connect-src` value:
 * `'self'`, scheme-sources (`http:`), and `scheme://host[:port]`
 * host-sources with an optional `*` port. Unknown tokens do not match.
 */
export function parseConnectSrcTokens(policy: string): string[] {
  return policy.trim().split(/\s+/).filter((token) => token !== "");
}

function wrapIpv6(hostname: string): string {
  return hostname.includes(":") ? `[${hostname}]` : hostname;
}

function unwrapIpv6(host: string): string {
  return host.startsWith("[") && host.endsWith("]") ? host.slice(1, -1) : host;
}

function effectivePort(url: URL): string {
  if (url.port) return url.port;

  return DEFAULT_PORTS[url.protocol.slice(0, -1)] ?? "";
}

/**
 * CSP scheme matching: same scheme, or the URL holds the secure upgrade of the
 * token's scheme (`https` of `http`, `wss` of `ws`); never a downgrade, and
 * never across the http/ws pairing — which is exactly why `http:` does not
 * cover the streaming sockets (B01).
 */
function schemeAllows(tokenScheme: string, urlScheme: string): boolean {
  if (tokenScheme === urlScheme) return true;

  return (
    (tokenScheme === "http" && urlScheme === "https") ||
    (tokenScheme === "ws" && urlScheme === "wss")
  );
}

/** `'self'`: same origin, or the same host and port over ws/wss as CSP defines it. */
function selfAllows(pageOrigin: string, url: URL): boolean {
  const origin = new URL(pageOrigin);

  if (origin.origin === url.origin) return true;

  return (
    (origin.protocol === "http:" || origin.protocol === "https:") &&
    (url.protocol === "ws:" || url.protocol === "wss:") &&
    origin.hostname === url.hostname &&
    effectivePort(origin) === effectivePort(url)
  );
}

const schemeSourcePattern = /^([a-z][a-z0-9+.-]*):$/i;

const hostSourcePattern = /^([a-z][a-z0-9+.-]*):\/\/(\[[^\]]+\]|[^\[\/:]+)(?::(\*|\d+))?$/i;

/**
 * Evaluate a `connect-src` token list against one URL the way the browser
 * would (for the token grammar above): true when any token permits the URL.
 * `pageOrigin` is the origin `'self'` resolves against.
 */
export function connectSrcAllows(
  tokens: ReadonlyArray<string>,
  url: string,
  pageOrigin: string,
): boolean {
  const target = new URL(url);

  for (const token of tokens) {
    if (token === "'self'") {
      if (selfAllows(pageOrigin, target)) return true;

      continue;
    }

    const schemeSource = schemeSourcePattern.exec(token)?.[1];

    if (schemeSource !== undefined) {
      if (schemeAllows(schemeSource.toLowerCase(), target.protocol.slice(0, -1))) return true;

      continue;
    }

    const hostSource = hostSourcePattern.exec(token);

    if (!hostSource) continue;

    const [, schemeGroup, hostGroup, tokenPort] = hostSource;

    if (schemeGroup === undefined || hostGroup === undefined) continue;

    const tokenScheme = schemeGroup;
    // Node's URL.hostname keeps IPv6 brackets ("[::1]"); unwrap both sides so
    // bracketed tokens and bracketed targets compare as the same host.
    const host = unwrapIpv6(hostGroup.toLowerCase());
    const targetHost = unwrapIpv6(target.hostname.toLowerCase());

    if (!schemeAllows(tokenScheme.toLowerCase(), target.protocol.slice(0, -1))) continue;

    if (host !== targetHost) continue;

    if (tokenPort === undefined) {
      if (effectivePort(target) === (DEFAULT_PORTS[tokenScheme.toLowerCase()] ?? "")) return true;

      continue;
    }

    if (tokenPort === "*" || tokenPort === effectivePort(target)) return true;
  }

  return false;
}
