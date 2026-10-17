"""PyMORGAN integration bridge for PyRATE-TA.

PyRATE-TA always depends on PyMORGAN for data structures (Dataset1D),
data loaders, font styling, and plotting backends. This module provides
detection utilities, metadata queries, and launcher hand-offs back to PyMORGAN.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from pyrate_ta.log import get_logger

logger = get_logger(__name__)


def get_pymorgan_info() -> dict[str, Any]:
    """Retrieve PyMORGAN installation metadata, version, and font status."""
    import pymorgan as pm

    info: dict[str, Any] = {
        "installed": True,
        "version": getattr(pm, "__version__", "unknown"),
        "package_dir": str(Path(pm.__file__).parent),
    }

    try:
        from pymorgan.fonts import are_fonts_installed

        info["fonts_configured"] = are_fonts_installed()
    except Exception:
        info["fonts_configured"] = False

    try:
        from pymorgan.settings import settings_path

        cfg = settings_path()
        info["settings_file"] = str(cfg) if cfg.exists() else None
    except Exception:
        info["settings_file"] = None

    return info


def find_pymorgan_command() -> list[str]:
    """Return command argv to execute PyMORGAN."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for name in ("PyMORGAN.exe", "pymorgan.exe", "PyMORGAN", "pymorgan"):
            cand = exe_dir / name
            if cand.is_file():
                return [str(cand)]
        for name in ("PyMORGAN_Suite.exe", "PyMORGAN_Suite"):
            cand = exe_dir / name
            if cand.is_file():
                return [str(cand), "--pymorgan"]

    py_exe = sys.executable
    if sys.platform == "win32":
        pyw = Path(py_exe).with_name("pythonw.exe")
        if pyw.is_file():
            py_exe = str(pyw)
    return [py_exe, "-m", "pymorgan"]


def launch_pymorgan(dataset_path: str | Path | None = None) -> subprocess.Popen:
    """Launch PyMORGAN in a detached background process with no console window.

    Parameters
    ----------
    dataset_path : str or Path, optional
        A dataset file path (e.g. .pdat, .h5) to load immediately in PyMORGAN.

    Returns
    -------
    subprocess.Popen
        The detached process handle.
    """
    cmd = list(find_pymorgan_command())
    if dataset_path is not None:
        p = Path(dataset_path)
        if p.exists():
            cmd.append(str(p))

    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    logger.info("Launching PyMORGAN: %s", " ".join(cmd))
    return subprocess.Popen(cmd, **kwargs)
