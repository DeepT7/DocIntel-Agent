"""Tests for input path resolution (Windows -> WSL conversion).

Run with::

    python -m unittest tests.test_scope -v
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.core import scope


def _kernel(release: str):
    return mock.patch.object(
        scope.platform, "uname", return_value=SimpleNamespace(release=release)
    )


class ResolveUserPathTest(unittest.TestCase):
    def test_windows_backslash_path_on_wsl(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"):
            self.assertEqual(
                scope.resolve_user_path(r"F:\LLM\DocIntel_Agent"),
                Path("/mnt/f/LLM/DocIntel_Agent"),
            )

    def test_windows_forward_slash_path_on_wsl(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"):
            self.assertEqual(
                scope.resolve_user_path("c:/Users/me/docs"),
                Path("/mnt/c/Users/me/docs"),
            )

    def test_quotes_are_stripped(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"):
            self.assertEqual(
                scope.resolve_user_path('"F:\\LLM\\Docs"'),
                Path("/mnt/f/LLM/Docs"),
            )

    def test_linux_path_unchanged_on_wsl(self):
        with _kernel("5.15.167.4-microsoft-standard-WSL2"):
            self.assertEqual(
                scope.resolve_user_path("/home/me/docs"),
                Path("/home/me/docs"),
            )

    def test_windows_path_not_converted_outside_wsl(self):
        with _kernel("6.8.0-generic"):
            path = scope.resolve_user_path(r"F:\LLM\Docs")
            self.assertNotEqual(path, Path("/mnt/f/LLM/Docs"))
            self.assertTrue(path.is_absolute())
