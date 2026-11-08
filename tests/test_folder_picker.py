"""Tests for the native folder picker wrapper (no dialog is opened)."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from app.server import folder_picker


def _kernel(release: str):
    return mock.patch.object(
        folder_picker.platform, "uname", return_value=SimpleNamespace(release=release)
    )


def _result(returncode: int, stdout: str = "", stderr: str = ""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


class PickFolderTest(unittest.TestCase):
    def test_wsl_uses_modern_windows_dialog(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"), mock.patch.object(
            folder_picker.shutil, "which", return_value="C:\\Windows\\powershell.exe"
        ), mock.patch.object(
            folder_picker.subprocess, "run", return_value=_result(0, "F:\\Docs")
        ) as run:
            self.assertEqual(folder_picker.pick_folder(), "F:\\Docs")
            command = run.call_args.args[0]
            self.assertEqual(command[0], "powershell.exe")
            self.assertIn("-STA", command)
            script = command[-1]
            self.assertIn("IFileOpenDialog", script)
            self.assertIn("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7", script)
            self.assertIn("0x20u", script)
            self.assertIn("0x40u", script)

    def test_native_windows_uses_powershell(self):
        with _kernel("10.0.22631"), mock.patch.object(
            folder_picker.os, "name", "nt"
        ), mock.patch.object(
            folder_picker.shutil, "which", return_value="C:\\Windows\\powershell.exe"
        ), mock.patch.object(
            folder_picker.subprocess, "run", return_value=_result(0, "C:\\Docs")
        ) as run:
            self.assertEqual(folder_picker.pick_folder(), "C:\\Docs")
            self.assertEqual(run.call_args.args[0][0], "powershell.exe")

    def test_other_platforms_use_tkinter(self):
        with _kernel("6.8.0-generic"), mock.patch.object(
            folder_picker.os, "name", "posix"
        ), mock.patch.object(
            folder_picker.subprocess, "run", return_value=_result(0, "/tmp/docs\n")
        ) as run:
            self.assertEqual(folder_picker.pick_folder(), "/tmp/docs")
            self.assertEqual(run.call_args.args[0][0], sys.executable)

    def test_wsl_without_powershell_raises(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"), mock.patch.object(
            folder_picker.shutil, "which", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "powershell"):
                folder_picker.pick_folder()

    def test_cancel_returns_none(self):
        with _kernel("6.8.0-generic"), mock.patch.object(
            folder_picker.os, "name", "posix"
        ), mock.patch.object(
            folder_picker.subprocess, "run", return_value=_result(0, "\n")
        ):
            self.assertIsNone(folder_picker.pick_folder())

    def test_failure_raises_with_last_stderr_line(self):
        with _kernel("6.8.0-generic"), mock.patch.object(
            folder_picker.os, "name", "posix"
        ), mock.patch.object(
            folder_picker.subprocess,
            "run",
            return_value=_result(1, stderr="warning\ntcl: no display name\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no display name"):
                folder_picker.pick_folder()

    def test_second_dialog_is_rejected(self):
        folder_picker._PICKER_LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already open"):
                folder_picker.pick_folder()
        finally:
            folder_picker._PICKER_LOCK.release()
