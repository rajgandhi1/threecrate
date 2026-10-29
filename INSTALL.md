# Install

Pick your agent below. Not listed? Any agent that supports [Agent Skills](https://agentskills.io) can use [`skills/threecrate`](skills/threecrate/SKILL.md): copy that folder into its skills folder (see [Any other agent](#any-other-agent)).

The skill drives the threecrate Python library, so `python` (3.8 or newer) needs to be on your PATH. The agent installs `threecrate` and `numpy` into a virtual environment the first time it uses the skill.

## Claude Code

```
/plugin marketplace add rajgandhi1/threecrate
```
```
/plugin install threecrate@threecrate
```
(Send these as two separate prompts.)

From your shell instead:

```bash
claude plugin marketplace add rajgandhi1/threecrate
claude plugin install threecrate@threecrate
```

Same steps in the Claude desktop app's Code tab: type the two `/plugin` commands into the prompt box.

The agent uses the skill on its own when a task involves point clouds or meshes. You can also call it directly with `/threecrate:threecrate`.

## Codex

```bash
codex plugin marketplace add rajgandhi1/threecrate
codex plugin add threecrate@threecrate
```

Start a new thread after installing. This also covers the Codex desktop app: restart the app and it picks up the plugin.

## Gemini CLI

```bash
gemini extensions install https://github.com/rajgandhi1/threecrate
```

Or install just the skill:

```bash
gemini skills install https://github.com/rajgandhi1/threecrate.git --path skills/threecrate
```

Run `/skills list` to check it loaded.

## OpenCode

OpenCode reads skills from `~/.config/opencode/skills/` (all projects) or `.opencode/skills/` (one project). It also reads `~/.agents/skills/` and `~/.claude/skills/`, so a skill installed for Claude Code or Codex works here too.

```bash
git clone --depth 1 https://github.com/rajgandhi1/threecrate.git
mkdir -p ~/.config/opencode/skills
cp -r threecrate/skills/threecrate ~/.config/opencode/skills/
```

## Cursor

Cursor reads skills from `~/.cursor/skills/` (all projects) or `.cursor/skills/` (one project), and also from `~/.agents/skills/`.

```bash
git clone --depth 1 https://github.com/rajgandhi1/threecrate.git
mkdir -p ~/.cursor/skills
cp -r threecrate/skills/threecrate ~/.cursor/skills/
```

Type `/` in Agent chat and search for `threecrate` to check it loaded.

## Any other agent

Most agents that support skills read the shared `~/.agents/skills/` folder. Copy the whole `skills/threecrate` folder there (not just `SKILL.md`; it includes task guides and a helper script):

```bash
git clone --depth 1 https://github.com/rajgandhi1/threecrate.git
mkdir -p ~/.agents/skills
cp -r threecrate/skills/threecrate ~/.agents/skills/
```

On Windows (PowerShell):

```powershell
git clone --depth 1 https://github.com/rajgandhi1/threecrate.git
New-Item -ItemType Directory -Force "$HOME\.agents\skills"
Copy-Item -Recurse threecrate\skills\threecrate "$HOME\.agents\skills\"
```

To share it with a team through one repository, copy it into the project's `.agents/skills/` instead and commit it.

## Try it

From the folder that holds your scans, ask:

- "Inspect scan.ply and tell me its size and point spacing"
- "Remove the noise from scan.ply and downsample it"
- "Align scan_b.ply to scan_a.ply and save the aligned cloud"
- "Find the floor in room.ply and save each object on it as its own file"
- "Turn bunny.ply into a smooth mesh"

## MCP server

Prefer tools over scripts? [threecrate-mcp](threecrate-mcp/README.md) gives agents the same abilities as MCP tools, and works in any MCP client. In Claude Code:

```bash
claude mcp add threecrate -- uvx threecrate-mcp
```

See its [README](threecrate-mcp/README.md) for Codex, OpenCode, Cursor and Claude Desktop.

## Update or remove

- **Claude Code:** `claude plugin update threecrate@threecrate` to update, `claude plugin uninstall threecrate@threecrate` to remove.
- **Codex:** `codex plugin marketplace upgrade threecrate` to update, `codex plugin remove threecrate@threecrate` to remove.
- **Gemini CLI:** `gemini extensions update threecrate` or `gemini extensions uninstall threecrate`.
- **Copied folders:** pull the repository again and copy the folder over the old one, or delete it.
