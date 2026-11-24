"""Tests for PyRATE-TA user target-scheme library, SchemeDialog library UI, and error guards."""

from __future__ import annotations

import logging
import os
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

import pyrate_ta as pr
from pyrate_ta.gui import app as pr_gui_app
from pyrate_ta.gui.scheme_dialog import SchemeDialog
from pyrate_ta.gui.tabs.fitting import FitTabMixin
from pyrate_ta.models.user_schemes import (
    UserSchemeEntry,
    delete_user_scheme,
    list_user_schemes,
    load_user_scheme,
    rename_user_scheme,
    save_user_scheme,
    user_schemes_dir,
)
from pyrate_ta.settings import settings_path

_VALID_SCHEME_TEXT = "S1 -> ICT : k1\nICT -> : k2\ninit S1 = 1\n"
_SECOND_SCHEME_TEXT = "A <-> B : k1, k2\nB -> C : k3\nC -> : k4\ninit A = 1\n"


@pytest.fixture(scope="module")
def qapp():
    """Ensure a single QApplication instance for offscreen Qt tests."""
    return QApplication.instance() or QApplication([])


# =========================================================================== #
# 1. user_schemes_dir() Priority Resolution                                   #
# =========================================================================== #


def test_user_schemes_dir_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv("PYRATE_SCHEMES_DIR", raising=False)
    settings = pr.get_settings()
    monkeypatch.setattr(settings, "schemes_dir", None)

    # 4. Default falls back to settings_path().parent / "schemes"
    assert user_schemes_dir() == (settings_path().parent / "schemes").resolve()

    # 3. Settings.schemes_dir overrides default
    cfg_dir = tmp_path / "from_settings"
    monkeypatch.setattr(settings, "schemes_dir", cfg_dir)
    assert user_schemes_dir() == cfg_dir.resolve()

    # 2. PYRATE_SCHEMES_DIR env var overrides Settings.schemes_dir
    env_dir = tmp_path / "from_env"
    monkeypatch.setenv("PYRATE_SCHEMES_DIR", str(env_dir))
    assert user_schemes_dir() == env_dir.resolve()

    # 1. Explicit override wins over all
    explicit_dir = tmp_path / "explicit"
    assert user_schemes_dir(explicit_dir) == explicit_dir.resolve()


# =========================================================================== #
# 2. Plain-Text CRUD, Validation & Path-Traversal Rejection                   #
# =========================================================================== #


def test_user_scheme_crud_validation_and_fallbacks(tmp_path, caplog):
    # Save valid scheme with overwrite protection
    saved = save_user_scheme("CT State Branching", _VALID_SCHEME_TEXT, directory=tmp_path)
    assert saved.is_file()
    assert saved.suffix == ".txt"
    content = saved.read_text(encoding="utf-8")
    assert content.startswith("# Name: CT State Branching\n")

    with pytest.raises(FileExistsError):
        save_user_scheme("CT State Branching", _SECOND_SCHEME_TEXT, directory=tmp_path)

    # overwrite=True updates text cleanly
    save_user_scheme(
        "CT State Branching", _SECOND_SCHEME_TEXT, overwrite=True, directory=tmp_path
    )
    assert "B -> C : k3" in saved.read_text(encoding="utf-8")

    # Plain .txt without # Name header uses filename stem
    (tmp_path / "plain_stem_model.txt").write_text(_VALID_SCHEME_TEXT, encoding="utf-8")
    # Invalid .txt file is skipped with warning, without crashing list_user_schemes
    (tmp_path / "broken_model.txt").write_text("not a valid reaction", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        entries = list_user_schemes(directory=tmp_path)

    assert "plain_stem_model" in entries
    assert entries["plain_stem_model"].name == "plain_stem_model"
    assert isinstance(entries["CT_State_Branching"], UserSchemeEntry)
    assert "broken_model" not in entries
    assert any("Skipping invalid user scheme broken_model.txt" in r.message for r in caplog.records)

    # Load by key or by display name
    by_key = load_user_scheme("CT_State_Branching", directory=tmp_path)
    by_name = load_user_scheme("CT State Branching", directory=tmp_path)
    assert by_key.n_rates == 4
    assert by_name.n_rates == 4

    # Rename updates path and # Name header
    renamed = rename_user_scheme(
        "CT State Branching", "Renamed Equilibrium Chain", directory=tmp_path
    )
    assert renamed.is_file()
    assert not saved.exists()
    assert load_user_scheme("Renamed Equilibrium Chain", directory=tmp_path).n_rates == 4

    # Delete removes existing file cleanly and returns False when absent
    assert delete_user_scheme("Renamed Equilibrium Chain", directory=tmp_path) is True
    assert not renamed.exists()
    assert delete_user_scheme("Renamed Equilibrium Chain", directory=tmp_path) is False

    # Empty and path-traversal names are rejected
    for bad_name in ("", "   ", "../escape", "sub/dir", r"sub\dir", "---"):
        with pytest.raises(ValueError):
            save_user_scheme(bad_name, _VALID_SCHEME_TEXT, directory=tmp_path)


# =========================================================================== #
# 3. pr.get_scheme() Built-in + User Scheme Fallback                          #
# =========================================================================== #


def test_get_scheme_builtin_and_user_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRATE_SCHEMES_DIR", str(tmp_path))
    monkeypatch.setattr(pr.get_settings(), "schemes_dir", None)

    save_user_scheme("Triplet Cascade", _SECOND_SCHEME_TEXT)

    builtin = pr.get_scheme("A_eq_B")
    assert builtin.key == "A_eq_B"

    by_key = pr.get_scheme("Triplet_Cascade")
    by_name = pr.get_scheme("Triplet Cascade")
    assert by_key.n_rates == 4
    assert by_name.n_rates == 4

    with pytest.raises(KeyError, match="Triplet_Cascade"):
        pr.get_scheme("no_such_scheme")


# =========================================================================== #
# 4. SchemeDialog Library UI Workflow (Save / Delete / Import / Export)       #
# =========================================================================== #


def test_scheme_dialog_library_workflow(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("PYRATE_SCHEMES_DIR", str(tmp_path))
    monkeypatch.setattr(pr.get_settings(), "schemes_dir", None)

    info_msgs = []
    warn_msgs = []
    crit_msgs = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: info_msgs.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: warn_msgs.append(a[2]))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: crit_msgs.append(a[2]))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes
    )

    dlg = SchemeDialog(None)
    try:
        # Built-in template entry disables Delete button
        assert dlg.TemplateCombo.count() >= 2
        dlg.TemplateCombo.setCurrentIndex(1)  # first built-in template
        assert dlg.TemplateCombo.currentData()[0] == "builtin"
        assert dlg.DeleteSchemeButton.isEnabled() is False

        # Save current valid text via _save_scheme
        dlg.SchemeText.setPlainText(_VALID_SCHEME_TEXT)
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("My Custom Model", True))
        dlg._save_scheme()

        assert (tmp_path / "My_Custom_Model.txt").is_file()
        assert "Saved model 'My Custom Model'" in dlg.StatusLabel.text()

        # Selecting the saved user scheme displays ★ prefix and enables Delete
        user_idx = -1
        for i in range(dlg.TemplateCombo.count()):
            data = dlg.TemplateCombo.itemData(i)
            if isinstance(data, tuple) and data == ("user", "My_Custom_Model"):
                user_idx = i
                break
        assert user_idx >= 0
        assert "★ My Custom Model" in dlg.TemplateCombo.itemText(user_idx)
        dlg.TemplateCombo.setCurrentIndex(user_idx)
        assert dlg.DeleteSchemeButton.isEnabled() is True
        assert "S1 -> ICT : k1" in dlg.SchemeText.toPlainText()

        # Export scheme to standalone .txt file
        export_path = tmp_path / "exported.txt"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **kw: (str(export_path), "")
        )
        dlg._export_scheme()
        assert export_path.is_file()
        assert "S1 -> ICT : k1" in export_path.read_text(encoding="utf-8")

        # Import another scheme from standalone .txt file and save copy to library
        incoming_dir = tmp_path / "incoming"
        incoming_dir.mkdir()
        import_src = incoming_dir / "external_model.txt"
        import_src.write_text(_SECOND_SCHEME_TEXT, encoding="utf-8")
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", lambda *a, **kw: (str(import_src), "")
        )
        dlg._import_scheme()
        assert "B -> C : k3" in dlg.SchemeText.toPlainText()
        assert (tmp_path / "external_model.txt").is_file()

        # Invalid scheme text on Save triggers warning instead of crashing
        dlg.SchemeText.setPlainText("invalid >>> syntax")
        dlg._save_scheme()
        assert any("fix syntax errors" in m.lower() for m in warn_msgs)

        # Corrupt/missing file on Import triggers QMessageBox.critical cleanly
        missing_src = tmp_path / "missing_file.txt"
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", lambda *a, **kw: (str(missing_src), "")
        )
        dlg._import_scheme()
        assert len(crit_msgs) >= 1

        # Delete the saved user scheme cleanly
        for i in range(dlg.TemplateCombo.count()):
            if dlg.TemplateCombo.itemData(i) == ("user", "My_Custom_Model"):
                dlg.TemplateCombo.setCurrentIndex(i)
                break
        dlg._delete_scheme()
        assert not (tmp_path / "My_Custom_Model.txt").exists()
    finally:
        dlg.close()


# =========================================================================== #
# 5. PyRATE-TA Error Guards & install_gui_excepthook                          #
# =========================================================================== #


def test_install_gui_excepthook_pyrate_ta(caplog):
    orig_sys = sys.excepthook
    orig_thr = threading.excepthook
    orig_default = sys.__excepthook__
    orig_pr_flag = pr_gui_app._IN_EXCEPTHOOK
    try:
        import pymorgan.gui.app as pm_gui_app

        orig_pm_flag = pm_gui_app._IN_EXCEPTHOOK
    except Exception:
        pm_gui_app = None
        orig_pm_flag = False

    kb_calls = []
    try:
        pr_gui_app.install_gui_excepthook("PyRATE-TA")

        with caplog.at_level(logging.ERROR):
            try:
                raise RuntimeError("pyrate gui boom")
            except RuntimeError as exc:
                sys.excepthook(type(exc), exc, exc.__traceback__)

        assert any("Unhandled exception in PyRATE-TA" in r.message for r in caplog.records)

        sys.__excepthook__ = lambda et, ev, tb: kb_calls.append(et)
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        assert kb_calls == [KeyboardInterrupt]
    finally:
        sys.__excepthook__ = orig_default
        sys.excepthook = orig_sys
        threading.excepthook = orig_thr
        pr_gui_app._IN_EXCEPTHOOK = orig_pr_flag
        if pm_gui_app is not None:
            pm_gui_app._IN_EXCEPTHOOK = orig_pm_flag


def test_fit_tab_edit_scheme_error_guard(qapp, monkeypatch):
    class _Stub(QWidget, FitTabMixin):
        pass

    crit_msgs = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: crit_msgs.append(a[2]))
    monkeypatch.setattr(
        "pyrate_ta.gui.scheme_dialog.edit_scheme",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("scheme dialog boom")),
    )

    stub = _Stub()
    try:
        stub.edit_scheme()
        assert any("scheme dialog boom" in m for m in crit_msgs)
    finally:
        stub.close()
