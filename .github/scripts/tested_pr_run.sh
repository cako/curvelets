#!/usr/bin/env bash
# Usage: tested_pr_run.sh REPO SHA BEFORE
#
# For a push to main, print the id of a CI run that already tested exactly
# this code, or nothing. That run is the latest pull_request run of the CI
# workflow for the head of the pull request that SHA merged, and it qualifies
# only if all of these hold:
#
# - the pull request was merged as SHA (merge, squash or rebase);
# - its head contained BEFORE, the previous tip of main (the branch was up to
#   date), so every pull_request run on that head tested the head as is;
# - SHA has the same tree as that head (the merge changed nothing);
# - the run succeeded and kept its coverage reports.
#
# Needs GH_TOKEN with read access to pull requests and actions. Prints
# nothing, and the caller runs the tests, if anything does not hold or any
# call fails.
set -euo pipefail

repo=$1
sha=$2
before=$3

log() { echo "$*" >&2; }

read -r number head < <(
  gh api "repos/$repo/commits/$sha/pulls" --jq \
    "map(select(.merged_at != null and .merge_commit_sha == \"$sha\"))
     | first | select(. != null) | \"\(.number) \(.head.sha)\""
) || { log "no pull request merged as $sha"; exit 0; }
log "merged pull request: #$number, head $head"

status=$(gh api "repos/$repo/compare/$before...$head" --jq .status)
if [[ $status != ahead && $status != identical ]]; then
  log "head does not contain the previous tip of main ($status): not up to date"
  exit 0
fi

head_tree=$(gh api "repos/$repo/commits/$head" --jq .commit.tree.sha)
sha_tree=$(gh api "repos/$repo/commits/$sha" --jq .commit.tree.sha)
if [[ $head_tree != "$sha_tree" ]]; then
  log "merged tree $sha_tree differs from the head tree $head_tree"
  exit 0
fi

read -r run conclusion < <(
  gh api "repos/$repo/actions/workflows/ci.yml/runs?event=pull_request&head_sha=$head&status=completed&per_page=1" \
    --jq '.workflow_runs | first | select(. != null) | "\(.id) \(.conclusion)"'
) || { log "no completed CI run for $head"; exit 0; }
if [[ $conclusion != success ]]; then
  log "latest CI run $run for $head: $conclusion"
  exit 0
fi

reports=$(gh api "repos/$repo/actions/runs/$run/artifacts?per_page=100" \
  --jq '[.artifacts[] | select((.name | startswith("coverage-")) and (.expired | not))] | length')
if ((reports == 0)); then
  log "CI run $run kept no coverage reports"
  exit 0
fi

log "CI run $run tested this code and kept $reports coverage reports"
echo "$run"
