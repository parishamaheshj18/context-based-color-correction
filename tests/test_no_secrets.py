"""Fails if a tracked file contains something that looks like an API key."""

import re
import subprocess
from pathlib import Path

import pytest

KEY_PATTERNS = [re.compile(rb"sk-ant-[A-Za-z0-9_-]{16,}"), re.compile(rb"sk-[A-Za-z0-9]{32,}")]
ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in output.decode().split("\0") if name]


def test_no_api_keys_in_tracked_files():
    leaks = [
        str(path.relative_to(ROOT))
        for path in _tracked_files()
        if path.is_file() and any(p.search(path.read_bytes()) for p in KEY_PATTERNS)
    ]
    assert not leaks, f"Possible API keys committed in: {leaks}"
