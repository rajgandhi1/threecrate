#!/usr/bin/env bash
set -euo pipefail

crate="${1:?crate name required}"
manifest_path="${2:-}"

args=(publish -p "$crate")
if [[ -n "$manifest_path" ]]; then
  args=(publish --manifest-path "$manifest_path")
fi

output=""
status=0
set +e
output="$(cargo "${args[@]}" 2>&1)"
status=$?
set -e

printf '%s\n' "$output"

if [[ $status -eq 0 ]]; then
  exit 0
fi

if grep -q "already exists on crates.io index" <<<"$output"; then
  echo "Skipping $crate: version already published"
  exit 0
fi

exit "$status"
