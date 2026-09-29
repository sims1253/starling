#!/usr/bin/env bash
set -euo pipefail

# A cancelled/failed upload leaves only a draft. Retries may fill that draft,
# but must never replace assets in a release that is already public.
if existing=$(gh release view "$RELEASE_TAG" --json isDraft --jq .isDraft 2>&1); then
  if [ "$existing" = false ]; then
    echo "Release $RELEASE_TAG is already published; keeping its original assets."
    exit 0
  fi
  if [ "$existing" != true ]; then
    echo "Unexpected isDraft value for $RELEASE_TAG: '$existing'" >&2
    exit 1
  fi
else
  case "$existing" in
    *"release not found"*|*"Release not found"*|*"HTTP 404"*)
      gh release create "$RELEASE_TAG" --target "$GITHUB_SHA" \
        --title "Experimental ${GITHUB_RUN_NUMBER} (${GITHUB_SHA:0:12})" \
        --draft --prerelease --latest=false \
        --notes-file "$RUNNER_TEMP/experimental-notes.md"
      ;;
    *) echo "Cannot determine state of release $RELEASE_TAG: $existing" >&2; exit 1 ;;
  esac
fi

gh release upload "$RELEASE_TAG" dist/* --clobber
gh release edit "$RELEASE_TAG" --draft=false --prerelease --latest=false \
  --notes-file "$RUNNER_TEMP/experimental-notes.md"
echo "Published https://github.com/$GH_REPO/releases/tag/$RELEASE_TAG" >> "$GITHUB_STEP_SUMMARY"
