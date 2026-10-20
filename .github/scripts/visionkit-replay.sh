#!/bin/bash
# Replay the Visionkit patch onto upstream release $TAG as local branch visionkit-$TAG.
# Used identically by the compile-check job and the push job, so the pushed branch is rebuilt from
# source by the push job itself, never taken from the job that ran the build.
#   TAG=3.32.3 SRC=visionkit-pin-3 bash .github/scripts/visionkit-replay.sh
set -euo pipefail
: "${TAG:?}" "${SRC:?}"
[[ "$TAG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "::error::unexpected tag '$TAG'"; exit 1; }
[[ "$SRC" =~ ^visionkit-(pin-[0-9]+|[0-9]+\.[0-9]+\.[0-9]+)$ ]] || { echo "::error::unexpected source branch '$SRC'"; exit 1; }
git remote get-url upstream > /dev/null 2>&1 || git remote add upstream https://github.com/ml-explore/mlx-swift-lm.git
git fetch --quiet upstream main --tags
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
# Commits on the patch branch that upstream doesn't already have (compared by content, so a patch that
# was upstreamed drops out instead of conflicting).
patches=$(git rev-list --reverse --no-merges --cherry-pick --right-only "upstream/main...origin/$SRC")
if [ -z "$patches" ]; then echo "::error::no Visionkit patch commits found on $SRC"; exit 1; fi
git checkout --quiet -B "visionkit-$TAG" "$TAG"
for c in $patches; do
  if ! git cherry-pick "$c"; then
    echo "::error::'$(git log -1 --format=%s "$c")' does not apply to $TAG: rebase the patch by hand"
    exit 1
  fi
done
git log --oneline "$TAG..HEAD"
