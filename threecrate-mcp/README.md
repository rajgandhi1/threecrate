# threecrate-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents work with 3D
point clouds and meshes using [threecrate](https://github.com/rajgandhi1/threecrate).
Ask your agent to "clean up scan.ply", "align these two scans" or "turn this cloud
into a mesh", and it calls these tools.

Works with any MCP client: Claude Code, Claude Desktop, Codex, OpenCode, Cursor,
Gemini CLI and others.

## Setup

You need [uv](https://docs.astral.sh/uv/) (for `uvx`) or Python 3.10+.

**Claude Code** (from your project folder):

```bash
claude mcp add threecrate -- uvx threecrate-mcp
```

**Codex** (from your project folder):

```bash
codex mcp add threecrate -- uvx threecrate-mcp
```

**OpenCode** (`opencode.json` in your project):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "threecrate": { "type": "local", "command": ["uvx", "threecrate-mcp"], "enabled": true }
  }
}
```

**Claude Desktop, Cursor and other apps** (their MCP JSON config). These apps
don't start servers in your project folder, so pass it with `--root`:

```json
{
  "mcpServers": {
    "threecrate": {
      "command": "uvx",
      "args": ["threecrate-mcp", "--root", "/path/to/your/scans"]
    }
  }
}
```

Without `uv`, install with `pip install threecrate-mcp` and use `threecrate-mcp`
as the command.

## How it behaves

- **One workspace folder.** Tools only read and write inside it: the folder the
  server starts in, or `--root` (also `THREECRATE_MCP_ROOT`). Paths are relative
  to it.
- **New files, never overwrites.** Results go to a new file next to the input
  (for example `scan.downsampled.ply`) unless you name one. Inputs are never
  overwritten, and existing files only with `overwrite=true`.
- **Summaries, not points.** Tools return counts, bounds, transforms and errors.
  Point data stays in files, so a big scan doesn't flood the agent's context.
- **Size limit.** Clouds above 50 million points are refused
  (`--max-points` or `THREECRATE_MCP_MAX_POINTS` to change).

## Tools

| Tool | What it does |
| --- | --- |
| `inspect_point_cloud` | Point count, bounds, spacing, and suggested voxel sizes and radii |
| `inspect_mesh` | Mesh health and size: holes, pieces, watertight, area, volume, printable or not (and why) |
| `convert` | Change format (PLY, PCD, XYZ, CSV; meshes PLY, OBJ, STL) |
| `downsample` | Voxel grid downsampling |
| `remove_outliers` | Statistical or radius outlier removal |
| `estimate_normals` | Save per-point normals to a PLY |
| `register` | Align two clouds: ICP, point-to-plane, GICP, or global (no starting pose needed) |
| `transform` | Apply a 4x4 rigid transform |
| `segment_plane` | Find a floor, wall or table plane; split the cloud |
| `extract_clusters` | Split a cloud into separate objects |
| `reconstruct` | Point cloud to mesh: auto, Poisson, ball pivoting, alpha shape, Delaunay |
| `simplify_mesh` | Reduce triangle count |
| `smooth_mesh` | Taubin, Laplacian or HC smoothing |
| `render` | Picture of one or more files (before and after) from several angles, returned to the agent; holes show in orange |
| `repair_for_print` | Make a scan or mesh printable (junk, floor, holes, missing base, face direction, size); writes STL and returns a before/after picture |

Readable point cloud formats: PLY, PCD, XYZ, CSV (and others threecrate
supports). Writable: PLY, PCD, XYZ, CSV.

## Agent Skill

If you would rather not run a server, the [threecrate skill](../skills/threecrate/SKILL.md)
teaches an agent to do the same work with short Python scripts. See
[INSTALL.md](../INSTALL.md) for Claude Code, Codex, Gemini CLI, OpenCode and
Cursor.

## Development

```bash
pip install -e ".[test]"
pytest
```
