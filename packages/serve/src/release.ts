/**
 * Release coordinate: npm package version ↔ GitHub release tag ↔ asset URL.
 *
 * The launcher downloads from the GitHub release whose tag matches the npm
 * package version (`starling-serve@0.2.0` ↔ release `v0.2.0`), so a published
 * package always resolves to the binaries built from the same source tree.
 */

export const DEFAULT_REPO = "sims1253/starling";

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
