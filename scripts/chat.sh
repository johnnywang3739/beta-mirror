#!/usr/bin/env bash
# Simple interactive chat loop using Claude Code + Ollama in pipe mode.
# Works in any terminal (no Ink TUI required).

set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
export CLAUDE_CODE_USE_OLLAMA=1
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:26b}"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Claude Code + Ollama ($OLLAMA_MODEL)            "
echo "║  Type your message, press Enter to send.         "
echo "║  Type 'exit' or Ctrl+C to quit.                  "
echo "╚══════════════════════════════════════════════════╝"
echo ""

while true; do
    printf "\033[1;36m❯ \033[0m"
    read -r input || break
    [ -z "$input" ] && continue
    [ "$input" = "exit" ] && echo "Bye!" && break
    [ "$input" = "quit" ] && echo "Bye!" && break

    echo ""
    echo "$input" | bun "$DIR/dist/cli.js" -p --dangerously-skip-permissions 2>/dev/null
    echo ""
done
