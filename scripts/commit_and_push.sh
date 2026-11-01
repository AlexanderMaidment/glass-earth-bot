#!/usr/bin/env bash
# Commit the given paths and push them to the branch this workflow runs on.
#
# The workflows that write forecast records call this as their last step:
#   bash scripts/commit_and_push.sh "<commit message>" <path> [<path> ...]
#
# It pulls with rebase and retries, so two runs that finish together both
# land. Ledger lines are only ever appended, and .gitattributes tells git to
# keep both sides' lines when two runs append at once.
set -euo pipefail

message="$1"
shift

# A rebase needs a committer too, so set the identity for every git command.
export GIT_AUTHOR_NAME="github-actions[bot]"
export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

for path in "$@"; do
  if [ -e "$path" ]; then
    git add -A -- "$path"
  fi
done

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  exit 0
fi

git commit --quiet -m "$message"

branch="${GITHUB_REF_NAME:-main}"
for attempt in 1 2 3 4 5; do
  if git pull --rebase --autostash --quiet origin "$branch" &&
    git push --quiet origin "HEAD:$branch"; then
    echo "Pushed to $branch."
    exit 0
  fi
  git rebase --abort 2>/dev/null || true
  echo "Push attempt $attempt failed. Trying again."
  sleep $((attempt * 10))
done

echo "Could not push after 5 attempts." >&2
exit 1
