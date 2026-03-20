"""Deterministic enforcement mechanisms for Stitch.

Generates hooks and configuration for tools that support infrastructure-level
enforcement (not just soft instructions). Currently supports:

1. Claude Code hooks (.claude/settings.json) — deterministic shell commands
2. Cursor .mdc rules with alwaysApply: true — reliably loaded every session

These mechanisms cannot be bypassed by the LLM choosing to skip instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def generate_claude_code_hooks() -> dict:
    """Generate Claude Code hook config for deterministic Stitch enforcement.

    The hooks use `hook-handler` which:
    - Reads the user's prompt from stdin JSON (provided by Claude Code)
    - Runs auto-setup + auto-route deterministically
    - Outputs context to stdout (Claude Code injects this into the conversation)

    Guard: `python3 -c 'import xstitch'` makes hooks no-ops on machines without Stitch.
    Suffix: `; true` ensures hooks never block the agent even if Stitch errors.
    """
    return {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            'python3 -c "import xstitch" 2>/dev/null && '
                            "python3 -m xstitch.cli hook-handler --event UserPromptSubmit; true"
                        ),
                    }
                ]
            }
        ],
        "PostToolUse": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            'python3 -c "import xstitch" 2>/dev/null && '
                            "python3 -m xstitch.cli hook-handler --event PostToolUse; true"
                        ),
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            'python3 -c "import xstitch" 2>/dev/null && '
                            "python3 -m xstitch.cli hook-handler --event Stop; true"
                        ),
                    }
                ]
            }
        ],
    }


def _merge_hooks(existing_hooks: dict, stitch_hooks: dict) -> dict:
    """Merge Stitch hooks into existing hook config.

    Replaces outdated Stitch hooks with current ones while preserving
    non-Stitch hooks from other tools.
    """
    for event, hook_list in stitch_hooks.items():
        if event not in existing_hooks:
            existing_hooks[event] = hook_list
        else:
            non_stitch = [
                h for h in existing_hooks[event]
                if "xstitch" not in json.dumps(h)
            ]
            existing_hooks[event] = non_stitch + hook_list
    return existing_hooks


def install_claude_code_hooks(dry_run: bool = False) -> str:
    """Install Stitch hooks into Claude Code's project-level settings.

    Replaces outdated Stitch hooks with current version while preserving
    non-Stitch hooks. Returns a description of what was done.
    """
    settings_dir = Path(".claude")
    settings_file = settings_dir / "settings.json"

    if dry_run:
        return f"Would install hooks in {settings_file}"

    settings_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    if settings_file.exists():
        try:
            raw = settings_file.read_text().strip()
            if raw:
                config = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            pass

    config["hooks"] = _merge_hooks(config.get("hooks", {}), generate_claude_code_hooks())
    settings_file.write_text(json.dumps(config, indent=2) + "\n")
    return f"Installed Stitch hooks in {settings_file}"


def install_claude_code_hooks_global(dry_run: bool = False) -> str:
    """Install Stitch hooks into Claude Code's global settings (~/.claude/settings.json)."""
    settings_file = Path.home() / ".claude" / "settings.json"

    if dry_run:
        return f"Would install global hooks in {settings_file}"

    settings_file.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if settings_file.exists():
        try:
            raw = settings_file.read_text().strip()
            if raw:
                config = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            pass

    config["hooks"] = _merge_hooks(config.get("hooks", {}), generate_claude_code_hooks())
    settings_file.write_text(json.dumps(config, indent=2) + "\n")
    return f"Installed Stitch global hooks in {settings_file}"


def check_claude_code_hooks() -> dict:
    """Check if Claude Code hooks are installed (project or global level)."""
    for label, path in [
        ("project", Path(".claude") / "settings.json"),
        ("global", Path.home() / ".claude" / "settings.json"),
    ]:
        if path.exists():
            try:
                config = json.loads(path.read_text())
                hooks = config.get("hooks", {})
                if any("xstitch" in json.dumps(hooks.get(e, [])) for e in ["UserPromptSubmit", "PostToolUse", "Stop"]):
                    return {"status": "ok", "detail": f"Hooks found in {label} settings ({path})"}
            except (json.JSONDecodeError, OSError):
                continue

    return {
        "status": "missing",
        "reason": "No Stitch hooks in Claude Code settings",
        "fix": "Run: stitch global-setup (or stitch doctor --fix)",
    }
