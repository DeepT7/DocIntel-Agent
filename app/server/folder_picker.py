"""Native folder picker for the local UI.

The browser cannot reveal an absolute folder path, so the local server opens the
OS dialog itself and returns the chosen path. Under WSL (and native Windows) the
Modern Explorer folder dialog is opened through PowerShell + IFileOpenDialog so
the user sees the normal Windows sidebar, breadcrumb and search. On other
platforms tkinter's directory dialog is used.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading

TIMEOUT_SECONDS = 120

_TK_SCRIPT = """
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
try:
    path = filedialog.askdirectory(parent=root, title="Select a folder", mustexist=True)
finally:
    root.destroy()
print(path or "")
"""

# IFileOpenDialog (CLSID DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7) with
# FOS_PICKFOLDERS (0x20) | FOS_FORCEFILESYSTEM (0x40). The dialog is owned by
# the foreground window so it opens on top of the browser.
_POWERSHELL_SCRIPT = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

[ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
class FileOpenDialog { }

[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IFileDialog
{
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint pfos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
    void AddPlace(IShellItem psi, int fdap);
    void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
    void Close(int hr);
    void SetClientGuid(ref Guid guid);
    void ClearClientData();
    void SetFilter(IntPtr pFilter);
}

[ComImport, Guid("d57c7288-d4ad-4768-be02-9d969532d960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IFileOpenDialog : IFileDialog { }

[ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IShellItem
{
    void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, out IntPtr ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
}

public static class DocIntelPicker
{
    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    public static string Show()
    {
        var dialog = (IFileOpenDialog)new FileOpenDialog();
        try
        {
            uint options;
            dialog.GetOptions(out options);
            dialog.SetOptions(options | 0x20u | 0x40u);
            dialog.SetTitle("Select a folder");
            int hr = dialog.Show(GetForegroundWindow());
            if (hr != 0) return null;

            IShellItem item;
            dialog.GetResult(out item);
            if (item == null) return null;

            IntPtr name;
            item.GetDisplayName(0x80058000u, out name);
            try
            {
                return Marshal.PtrToStringUni(name);
            }
            finally
            {
                if (name != IntPtr.Zero) Marshal.FreeCoTaskMem(name);
            }
        }
        finally
        {
            Marshal.ReleaseComObject(dialog);
        }
    }
}
"@
$path = [DocIntelPicker]::Show()
if ($path) { [Console]::Out.Write($path) }
"""

_PICKER_LOCK = threading.Lock()


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _uses_windows_dialog() -> bool:
    return _is_wsl() or os.name == "nt"


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"folder picker failed: {error}") from error

    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "folder picker failed")

    return result.stdout.strip() or None


def pick_folder() -> str | None:
    """Open the native folder dialog and return the path (``None`` if cancelled)."""
    if not _PICKER_LOCK.acquire(blocking=False):
        raise RuntimeError("a folder dialog is already open")
    try:
        if _uses_windows_dialog():
            if shutil.which("powershell.exe") is None:
                raise RuntimeError(
                    "powershell.exe not found; enable WSL interop or type the path"
                )
            return _run(
                ["powershell.exe", "-NoProfile", "-STA", "-Command", _POWERSHELL_SCRIPT]
            )
        return _run([sys.executable, "-c", _TK_SCRIPT])
    finally:
        _PICKER_LOCK.release()
