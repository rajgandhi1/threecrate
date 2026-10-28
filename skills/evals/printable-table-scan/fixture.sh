#!/usr/bin/env bash
# Copy the sample scan into the empty workspace.
set -euo pipefail
cp "$(dirname "$0")/../../../samples/table_scan.ply" .
