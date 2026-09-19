#!/usr/bin/env bash
set -eu

instructions_dir="/home/vscode/.copilot/instructions"
global_claude_file="/home/vscode/.claude/CLAUDE.md"
global_copilot_file="$instructions_dir/global-claude.instructions.md"
settings_instruction_file="$instructions_dir/claude-settings.instructions.md"

mkdir -p "$instructions_dir"

# Keep Copilot's global instructions synchronized with the shared Claude file.
ln -sfn "$global_claude_file" "$global_copilot_file"

cat > "$settings_instruction_file" <<'EOF'
# Project lifecycle settings

The repository's notebook lifecycle hook is
`.codex/hooks/sync_analysis_py.py`. Claude Code and Codex run equivalent
project hooks automatically. Copilot CLI must use the same hook explicitly:
before inspecting or editing repository files, run

```bash
printf '{"cwd":"%s"}\n' "$PWD" | /usr/bin/python3 .codex/hooks/sync_analysis_py.py
```

Run it from the repository root. It updates a paired `.py` mirror only when
the corresponding `.ipynb` file is newer. Do not edit paired `.ipynb` files
directly; edit the percent-format `.py` source and synchronize it back with
`uv run jupytext --sync <source>.py` after changes.

Preserve the repository's existing Claude and Codex hooks, permissions, and
other project conventions. This instruction is the Copilot-compatible bridge
to those project hooks; it does not make Copilot execute Claude settings
automatically.
EOF

chown -R vscode:vscode /home/vscode/.copilot
