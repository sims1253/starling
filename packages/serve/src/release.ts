/**
 * Release coordinate: npm package version ↔ GitHub release tag ↔ asset URL.
 *
 * The launcher downloads from the GitHub release whose tag matches the npm
 * package version (`starling-serve@0.2.0` ↔ release `v0.2.0`), so a published
 * package always resolves to the binaries built from the same source tree.
 */
import { assertCacheComponent, InvalidCacheComponentError } from "./cache.js";

export const DEFAULT_REPO = "sims1253/starling";

/**
 * Normalize a `owner/name` repository coordinate for cache identity: trim
 * surrounding whitespace and lowercase. GitHub resolves owner and repository
 * names case-insensitively, so both spellings denote the same release source
 * and must share one cache entry — otherwise one spelling's binary could
 * shadow the other's. Malformed or path-unsafe coordinates throw
 * {@link InvalidCacheComponentError} before any download is attempted.
 */
export function normalizeRepo(repo: string): string {
  const [owner, name, ...extra] = repo.trim().split("/");

  if (
    owner === undefined ||
    owner === "" ||
    name === undefined ||
    name === "" ||
    extra.length > 0
  ) {
    throw new InvalidCacheComponentError("repository", repo);
  }

  const normalizedOwner = owner.toLowerCase();
  const normalizedName = name.toLowerCase();
  assertCacheComponent("repository owner", normalizedOwner);
  assertCacheComponent("repository name", normalizedName);

  return `${normalizedOwner}/${normalizedName}`;
}

/** Normalize a tag with or without the leading `v` to the `vX.Y.Z` form. */
export function normalizeTag(tag: string): string {
  return tag.startsWith("v") ? tag : `v${tag}`;
}

/** The release tag a package version maps to, honoring an explicit override. */
export function releaseTag(version: string, override?: string): string {
  return normalizeTag(override ?? version);
}

export function releaseAssetUrl(repo: string, tag: string, asset: string): string {
  return `https://github.com/${repo}/releases/download/${normalizeTag(tag)}/${asset}`;
}
