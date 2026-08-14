"""Open ``main_window.ui`` in Qt Designer (``python -m pyrate.gui.designer``).

Every widget, layout and control must be declared in the ``.ui`` file rather
than built in Python, so this is the supported way to change the

layout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    ui_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("main_window.ui")
    cmd = ["uvx", "--from", "pyside6-essentials", "pyside6-designer", str(ui_path)]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception as exc:
        print(f"Error running command: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
