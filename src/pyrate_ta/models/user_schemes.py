"""Persistent user library of target-analysis kinetic schemes stored as plain text.

Each user model lives as a UTF-8 ``.txt`` file inside :func:`user_schemes_dir`
(defaulting to ``~/.config/pyrate_ta/schemes/``, or overridden via the
``PYRATE_SCHEMES_DIR`` environment variable or ``Settings.schemes_dir``).
Files use the exact :mod:`~pyrate_ta.models.scheme_text` notation so they can
be tracked in Git and shared across lab members, with an optional
``# Name: <Display Name>`` header comment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..log import get_logger
from .scheme_text import SchemeSyntaxError, parse_scheme_text, scheme_from_text

logger = get_logger(__name__)

_NAME_HEADER_RE = re.compile(r"^\s*#\s*name\s*:\s*(.+?)\s*$", re.IGNORECASE)
_UNSAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class UserSchemeEntry:
    """Metadata and raw notation text for a saved user kinetic scheme."""

    key: str
    name: str
    path: Path
    text: str


def user_schemes_dir(override: str | Path | None = None) -> Path:
    """Return the directory where custom plain-text target schemes are stored.

    Resolution order:
    1. Explicit ``override`` path.
    2. ``PYRATE_SCHEMES_DIR`` environment variable (when non-empty).
    3. ``Settings.schemes_dir`` (when configured).
    4. ``settings_path().parent / "schemes"`` (defaults to
       ``~/.config/pyrate_ta/schemes``).
    """
    if override is not None:
        return Path(override).expanduser().resolve()

    env_dir = os.environ.get("PYRATE_SCHEMES_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    from ..settings import get_settings, settings_path

    cfg_dir = getattr(get_settings(), "schemes_dir", None)
    if cfg_dir:
        return Path(cfg_dir).expanduser().resolve()

    return (settings_path().parent / "schemes").resolve()


def _slugify_scheme_name(name: str) -> str:
    """Convert a display name into a safe single-segment filename stem."""
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Scheme name cannot be empty.")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"Scheme name {raw!r} must not contain path separators or '..'.")
    slug = _UNSAFE_SLUG_RE.sub("_", raw).strip("._-")
    if not slug:
        raise ValueError(
            f"Scheme name {raw!r} must contain at least one alphanumeric character."
        )
    return slug


def _extract_display_name(text: str, fallback_stem: str) -> str:
    """Read ``# Name: ...`` from the header comments, falling back to ``fallback_stem``."""
    for line in str(text).splitlines()[:5]:
        match = _NAME_HEADER_RE.match(line)
        if match:
            val = match.group(1).strip()
            if val:
                return val
    return fallback_stem


def _with_name_header(name: str, text: str) -> str:
    """Embed or update the ``# Name: <name>`` comment line in ``text``."""
    clean_name = " ".join(str(name).splitlines()).strip()
    lines = str(text).splitlines()
    if lines and _NAME_HEADER_RE.match(lines[0]):
        lines[0] = f"# Name: {clean_name}"
    else:
        lines.insert(0, f"# Name: {clean_name}")
    return "\n".join(lines).rstrip() + "\n"


def list_user_schemes(directory: str | Path | None = None) -> dict[str, UserSchemeEntry]:
    """Return all valid saved ``.txt`` schemes keyed by filename stem (case-insensitive sorted)."""
    base = user_schemes_dir(directory)
    if not base.is_dir():
        return {}

    entries: list[UserSchemeEntry] = []
    for path in sorted(base.glob("*.txt"), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            parse_scheme_text(text)
        except (OSError, SchemeSyntaxError, ValueError) as exc:
            logger.warning("Skipping invalid user scheme %s: %s", path.name, exc)
            continue
        key = path.stem
        display_name = _extract_display_name(text, key)
        entries.append(UserSchemeEntry(key=key, name=display_name, path=path, text=text))

    entries.sort(key=lambda e: (e.name.lower(), e.key.lower()))
    return {entry.key: entry for entry in entries}


def _find_entry(
    name_or_key: str, directory: str | Path | None = None
) -> UserSchemeEntry | None:
    entries = list_user_schemes(directory)
    target = str(name_or_key or "").strip()
    if not target:
        return None
    if target in entries:
        return entries[target]
    low = target.lower()
    for entry in entries.values():
        if entry.key.lower() == low or entry.name.lower() == low:
            return entry
    try:
        slug = _slugify_scheme_name(target).lower()
    except ValueError:
        return None
    for entry in entries.values():
        if entry.key.lower() == slug:
            return entry
    return None


def load_user_scheme(name_or_key: str, directory: str | Path | None = None):
    """Load and parse a saved user scheme into a :class:`~pyrate_ta.models.schemes.TargetScheme`."""
    entry = _find_entry(name_or_key, directory=directory)
    if entry is None:
        avail = ", ".join(sorted(list_user_schemes(directory))) or "(none)"
        raise KeyError(
            f"unknown user target scheme {name_or_key!r}; available user schemes: {avail}"
        )
    return scheme_from_text(entry.text, key=entry.key)


def save_user_scheme(
    name: str,
    text: str,
    *,
    overwrite: bool = False,
    directory: str | Path | None = None,
) -> Path:
    """Validate ``text`` and save it under ``name`` as a plain-text ``.txt`` file."""
    clean_name = str(name or "").strip()
    slug = _slugify_scheme_name(clean_name)
    # Validate notation before creating directories or touching disk.
    scheme_from_text(text, key=slug)

    base = user_schemes_dir(directory)
    target = base / f"{slug}.txt"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Saved target scheme {clean_name!r} already exists at {target}")

    base.mkdir(parents=True, exist_ok=True)
    target.write_text(_with_name_header(clean_name, text), encoding="utf-8")
    logger.info("Saved user target scheme %r to %s", clean_name, target)
    return target


def delete_user_scheme(name_or_key: str, directory: str | Path | None = None) -> bool:
    """Delete a saved user scheme file. Returns ``True`` if a file was removed."""
    base = user_schemes_dir(directory)
    if not base.is_dir():
        return False

    entry = _find_entry(name_or_key, directory=base)
    if entry is not None and entry.path.is_file():
        entry.path.unlink()
        logger.info("Deleted user target scheme %r (%s)", entry.name, entry.path)
        return True

    # Also allow deleting an unparseable file when addressed directly by valid stem.
    try:
        slug = _slugify_scheme_name(name_or_key)
    except ValueError:
        return False
    direct = base / f"{slug}.txt"
    if direct.is_file():
        direct.unlink()
        logger.info("Deleted user target scheme file %s", direct)
        return True
    return False


def rename_user_scheme(
    old_name: str,
    new_name: str,
    directory: str | Path | None = None,
) -> Path:
    """Rename an existing saved user scheme cleanly on disk."""
    base = user_schemes_dir(directory)
    entry = _find_entry(old_name, directory=base)
    if entry is None:
        raise KeyError(f"unknown user target scheme {old_name!r}")

    clean_new = str(new_name or "").strip()
    new_slug = _slugify_scheme_name(clean_new)
    new_path = base / f"{new_slug}.txt"
    if new_path != entry.path and new_path.exists():
        raise FileExistsError(f"Saved target scheme {clean_new!r} already exists at {new_path}")

    updated = _with_name_header(clean_new, entry.text)
    new_path.write_text(updated, encoding="utf-8")
    if new_path != entry.path and entry.path.is_file():
        entry.path.unlink()
    logger.info("Renamed user target scheme %r -> %r (%s)", entry.name, clean_new, new_path)
    return new_path
