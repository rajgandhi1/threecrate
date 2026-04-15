#!/usr/bin/env bash
# Captures screenshots of threecrate examples into assets/screenshots/
# Usage:
#   ./scripts/capture_screenshots.sh          # all examples
#   ./scripts/capture_screenshots.sh one      # interactive_viewer_example only (for testing)

set -e
OUTPUT_DIR="assets/screenshots"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Find the CGWindowID for a window owned by the given PID.
# Retries for up to $timeout seconds. Prints the window ID, or "" on failure.
# Uses an inline Swift script — no extra dependencies needed.
# ---------------------------------------------------------------------------
WINDOW_FINDER_SWIFT="$(mktemp /tmp/winfinder.XXXXXX.swift)"
cat > "$WINDOW_FINDER_SWIFT" <<'SWIFT'
import CoreGraphics
import Foundation

guard CommandLine.arguments.count == 2,
      let targetPID = Int32(CommandLine.arguments[1]) else {
    exit(1)
}

let options = CGWindowListOption.optionOnScreenOnly
guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
    exit(1)
}

for info in list {
    guard let pid = info[kCGWindowOwnerPID as String] as? Int32,
          pid == targetPID,
          let layer = info[kCGWindowLayer as String] as? Int,
          layer == 0,                        // normal app window, not menu bar etc.
          let winID = info[kCGWindowNumber as String] as? UInt32,
          winID > 0 else { continue }
    print(winID)
    exit(0)
}
exit(1)
SWIFT

# Also compile a "list all windows" helper for diagnostics
WINDOW_LIST_SWIFT="$(mktemp /tmp/winlist.XXXXXX.swift)"
cat > "$WINDOW_LIST_SWIFT" <<'SWIFT'
import CoreGraphics
import Foundation

let options = CGWindowListOption.optionOnScreenOnly
guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
    exit(1)
}

for info in list {
    guard let layer = info[kCGWindowLayer as String] as? Int, layer == 0 else { continue }
    let owner = info[kCGWindowOwnerName as String] as? String ?? "(no owner)"
    let name  = info[kCGWindowName  as String] as? String ?? "(no name)"
    let pid   = info[kCGWindowOwnerPID as String] as? Int32 ?? 0
    print("  [pid=\(pid)] [\(owner)] '\(name)'")
}
SWIFT

cleanup() {
    rm -f "$WINDOW_FINDER_SWIFT" "$WINDOW_LIST_SWIFT"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait for a layer-0 window owned by $bin_pid to appear.
# Prints the CGWindowID on success, "" on timeout.
# ---------------------------------------------------------------------------
wait_for_window_id() {
  local bin_pid="$1"
  local timeout="${2:-20}"
  local elapsed=0

  while [ "$elapsed" -lt "$timeout" ]; do
    local win_id
    win_id=$(swift "$WINDOW_FINDER_SWIFT" "$bin_pid" 2>/dev/null) || true
    if [ -n "$win_id" ]; then
      echo "$win_id"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo ""
}

# ---------------------------------------------------------------------------
# Run a visual example, find its window by PID, screenshot it, kill it.
# $3 = optional extra cargo flags, e.g. "--features bevy_interop"
# ---------------------------------------------------------------------------
capture_visual() {
  local example="$1"
  local wait_secs="${2:-20}"
  local extra_flags="${3:-}"
  local out="$OUTPUT_DIR/${example}.png"

  echo ""
  echo "▶ $example"

  # shellcheck disable=SC2086
  cargo run -p threecrate-examples --bin "$example" $extra_flags &
  local cargo_pid=$!

  # Give cargo time to link and exec the binary
  sleep 4

  # Resolve the actual binary PID (child of cargo)
  local bin_pid
  bin_pid=$(pgrep -f "target/debug/$example" | head -1) || true

  if [ -z "$bin_pid" ]; then
    echo "  ERROR: binary '$example' process not found after 4s — did it crash?"
    kill "$cargo_pid" 2>/dev/null || true
    wait "$cargo_pid" 2>/dev/null || true
    return 1
  fi

  echo "  binary pid=$bin_pid, waiting up to ${wait_secs}s for window..."

  local win_id
  win_id=$(wait_for_window_id "$bin_pid" "$wait_secs")

  if [ -z "$win_id" ]; then
    echo "  ERROR: no window found for pid=$bin_pid after ${wait_secs}s"
    echo "  All visible layer-0 windows at timeout:"
    swift "$WINDOW_LIST_SWIFT" 2>/dev/null || echo "  (swift failed)"
    kill "$cargo_pid" 2>/dev/null || true
    wait "$cargo_pid" 2>/dev/null || true
    return 1
  fi

  echo "  found window id=$win_id — waiting for first rendered frame..."

  # Bring window to front before capture
  osascript 2>/dev/null <<EOF || true
tell application "System Events"
  set frontmost of first process whose unix id is $bin_pid to true
end tell
EOF
  # Give Bevy time to finish startup systems, generate meshes/point clouds,
  # upload GPU data, and present the first real frame before we capture.
  sleep 4

  screencapture -x -o -l"$win_id" "$out"
  echo "  ✓ saved → $out"

  kill "$cargo_pid" 2>/dev/null || true
  wait "$cargo_pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Visual examples
# ---------------------------------------------------------------------------
run_visual_examples() {
  capture_visual "bevy_mesh_viewer"            30  "--features bevy_interop"
  capture_visual "bevy_conversion_demo"        30  "--features bevy_interop"
  capture_visual "interactive_viewer_example"  30  "--features bevy_interop"
  capture_visual "gpu_mesh_render_example"     30  "--features bevy_interop"
  # pbr_visualization skipped — crashes with wgpu buffer size mismatch (see issue #133)
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if [ "${1}" = "one" ]; then
  echo "=== Test: bevy_mesh_viewer ==="
  capture_visual "bevy_mesh_viewer" 30 "--features bevy_interop"
  echo ""
  echo "=== Done — check $OUTPUT_DIR/bevy_mesh_viewer.png ==="
else
  echo "=== Building examples... ==="
  cargo build -p threecrate-examples
  cargo build -p threecrate-examples --features bevy_interop

  echo ""
  echo "=== Visual examples ==="
  run_visual_examples

  echo ""
  echo "=== All done. Review files in $OUTPUT_DIR/ ==="
fi
