#!/usr/bin/env bash
# One-line install (official tree — use your fork URL if you forked):
#   curl -fsSL https://raw.githubusercontent.com/claude-code-best/claude-code/main/scripts/install-cli.sh | bash
#
# Environment (optional):
#   CLAUDE_CODE_REPO   Git clone URL (default: upstream repo below)
#   CLAUDE_CODE_BRANCH Branch to clone (default: main)
#   CLAUDE_CODE_DIR    Install directory when cloning (default: ~/.local/share/claude-code-src)
#   SKIP_BUN_INSTALL    Set to 1 to skip installing Bun when missing
#   ADD_BUN_TO_PATH     Set to 1 to append export PATH=...bun/bin to ~/.bashrc and ~/.zshrc if missing

set -euo pipefail

DEFAULT_REPO="${CLAUDE_CODE_REPO:-https://github.com/claude-code-best/claude-code.git}"
DEFAULT_BRANCH="${CLAUDE_CODE_BRANCH:-main}"
INSTALL_ROOT="${CLAUDE_CODE_DIR:-$HOME/.local/share/claude-code-src}"

need() { command -v "$1" >/dev/null 2>&1; }

ensure_bun() {
  if need bun; then
    return 0
  fi
  if [[ "${SKIP_BUN_INSTALL:-0}" == "1" ]]; then
    echo "error: bun is not installed and SKIP_BUN_INSTALL=1" >&2
    exit 1
  fi
  echo "=> Installing Bun (https://bun.sh) ..."
  curl -fsSL https://bun.sh/install | bash
  export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
  export PATH="$BUN_INSTALL/bin:$PATH"
}

append_path_hint() {
  local line='export PATH="$HOME/.bun/bin:$PATH"'
  if [[ "${ADD_BUN_TO_PATH:-0}" != "1" ]]; then
    echo "=> Add Bun to your PATH (once per machine), then open a new terminal:"
    echo "     $line"
    return 0
  fi
  for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    [[ -f "$rc" ]] || continue
    if grep -qF '.bun/bin' "$rc" 2>/dev/null; then
      echo "=> $rc already mentions .bun/bin — skipping"
      continue
    fi
    echo "" >>"$rc"
    echo "# Bun (added by claude-code install-cli.sh)" >>"$rc"
    echo "$line" >>"$rc"
    echo "=> Appended PATH line to $rc"
  done
}

resolve_src_dir() {
  # Running from a git checkout: .../claude-code/scripts/install-cli.sh
  local here="${BASH_SOURCE[0]:-$0}"
  if [[ -n "$here" && "$here" != bash && "$here" != "-" && -f "$here" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "$here")" && pwd)"
    local root
    root="$(cd "$script_dir/.." && pwd)"
    if [[ -f "$root/package.json" ]] && grep -q '"name"[[:space:]]*:[[:space:]]*"claude-js"' "$root/package.json" 2>/dev/null; then
      echo "$root"
      return 0
    fi
  fi
  echo ""
}

main() {
  ensure_bun
  export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
  export PATH="$BUN_INSTALL/bin:$PATH"

  local src
  src="$(resolve_src_dir)"
  if [[ -z "$src" ]]; then
    need git || {
      echo "error: git is required to clone $DEFAULT_REPO" >&2
      exit 1
    }
    echo "=> Cloning or updating repo under $INSTALL_ROOT ..."
    mkdir -p "$(dirname "$INSTALL_ROOT")"
    if [[ -d "$INSTALL_ROOT/.git" ]]; then
      git -C "$INSTALL_ROOT" fetch origin "$DEFAULT_BRANCH" --depth 1 2>/dev/null || true
      git -C "$INSTALL_ROOT" checkout "$DEFAULT_BRANCH" 2>/dev/null || git -C "$INSTALL_ROOT" checkout -B "$DEFAULT_BRANCH" "origin/$DEFAULT_BRANCH"
      git -C "$INSTALL_ROOT" pull --ff-only origin "$DEFAULT_BRANCH" || true
    else
      rm -rf "$INSTALL_ROOT"
      git clone --depth 1 --branch "$DEFAULT_BRANCH" "$DEFAULT_REPO" "$INSTALL_ROOT"
    fi
    src="$INSTALL_ROOT"
  else
    echo "=> Using existing checkout: $src"
  fi

  cd "$src"
  echo "=> bun install"
  bun install
  echo "=> bun run build"
  bun run build
  echo "=> bun link (installs claude-js into ~/.bun/bin)"
  bun link

  append_path_hint
  echo ""
  echo "Done. Try:  claude-js --help"
  echo "Ollama:     CLAUDE_CODE_USE_OLLAMA=1 OLLAMA_MODEL=gemma4:26b claude-js"
}

main "$@"
