#!/usr/bin/env python3
"""Adaptive test runner for PyRATE-TA.

Detects the files modified in the git working tree and runs only the relevant
subset of tests. Falls back to the full suite when ``--all`` is passed, when a
global/configuration file changed, or when a change cannot be mapped to a
suite -- an unmapped change must never be reported as "tested".
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# Source path prefix -> the test suites that cover it.
RULES: list[tuple[str, list[str]]] = [
    ("src/pyrate/gui/", ["tests/test_gui_ui.py", "tests/test_gui.py"]),
    ("src/pyrate/models/", ["tests/test_models.py"]),
    ("src/pyrate/fit/", ["tests/test_fit.py", "tests/test_recovery.py"]),
    ("src/pyrate/results/", ["tests/test_results.py"]),
    ("src/pyrate/plot/", ["tests/test_plot.py"]),
    ("src/pyrate/io/", ["tests/test_io.py"]),
    ("src/pyrate/settings.py", ["tests/test_settings.py"]),
    ("src/pyrate/log.py", ["tests/test_scaffold.py"]),
]

# A change to any of these invalidates the mapping above.
GLOBAL_FILES = {
    "pyproject.toml",
    "src/pyrate/__init__.py",
    "src/pyrate/__about__.py",
    "tests/conftest.py",
    "tests/synthetic.py",
}


def get_changed_files() -> list[str] | None:
    """Files modified in the index or working tree; ``None`` outside a repo."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        print(f"Warning: could not run git status (not a git repository?): {exc}")
        return None

    files = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:  # renamed: take the destination
            path = path.split(" -> ")[1].strip()
        files.append(path)
    return files


def resolve_tests(changed_files: list[str]) -> set[str]:
    """Map changed paths to test files; an empty set means "run everything"."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    tests: set[str] = set()

    for f in changed_files:
        f_norm = f.replace("\\", "/")

        if f_norm in GLOBAL_FILES:
            print(f"Global file modified: {f_norm}. Running the full suite.")
            return set()

        if f_norm.startswith("tests/test_") and f_norm.endswith(".py"):
            tests.add(f_norm)
            continue

        matched = False
        for prefix, test_list in RULES:
            if f_norm.startswith(prefix):
                # Suites that do not exist yet are simply skipped: the mapping
                # is written ahead of the modules it will cover.
                tests.update(t for t in test_list if (repo / t).is_file())
                matched = True

        if not matched and (f_norm.startswith("src/") or f_norm.startswith("tests/")):
            print(f"Unmapped change in the source/test tree: {f_norm}. Running the full suite.")
            return set()

    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive test runner for PyRATE-TA.")
    parser.add_argument("--all", action="store_true", help="Force the full test suite.")
    args, pytest_args = parser.parse_known_args()

    if args.all:
        print("Forcing a full test-suite run.")
        return subprocess.run([sys.executable, "-m", "pytest", *pytest_args]).returncode

    changed = get_changed_files()
    if changed is None:
        print("Not a git repository. Running the full test suite.")
        return subprocess.run([sys.executable, "-m", "pytest", *pytest_args]).returncode

    if not changed:
        print("No changed files. Running the smoke checks (run_checks.py).")
        return subprocess.run([sys.executable, "scripts/run_checks.py"]).returncode

    targets = resolve_tests(changed)
    if not targets:
        print("The changes require the full test suite.")
        cmd = [sys.executable, "-m", "pytest", *pytest_args]
    else:
        print(f"Running tests for the changed files: {sorted(targets)}")
        cmd = [sys.executable, "-m", "pytest", *sorted(targets), *pytest_args]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
