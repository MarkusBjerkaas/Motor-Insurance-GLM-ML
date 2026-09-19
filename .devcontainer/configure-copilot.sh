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
# Claude project settings

Treat `.claude/settings.json` as the authoritative project configuration for
Claude Code. Preserve and follow the repository's existing hooks, permissions,
and other project conventions when making changes. Copilot cannot execute
Claude Code settings directly, so use this file as project context rather than
trying to translate it into Copilot tool permissions.
EOF

chown -R vscode:vscode /home/vscode/.copilot
