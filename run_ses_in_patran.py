"""Automate Patran 2024 session file (.ses.01) execution.

Flow:
  1. Focus Patran and paste !!input "filepath" into the command bar.
  2. Monitor for "Message" popups and auto-click Yes.
  3. Exit early once no popups appear for --idle seconds.

Usage:
    python run_ses_in_patran.py wing_constant_profile.ses.01
    python run_ses_in_patran.py wing_cosine_spacing.ses.01 --timeout 180
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import pyautogui
import pygetwindow as gw
import pyperclip

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05  # minimal pause for speed

_user32 = ctypes.windll.user32


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def find_patran_window() -> gw.Win32Window | None:
    for win in gw.getAllWindows():
        if "patran" in win.title.lower() and win.visible:
            return win
    return None


def find_message_popup() -> gw.Win32Window | None:
    for win in gw.getAllWindows():
        if win.visible and win.title.lower() == "message":
            return win
    return None


# ------------------------------------------------------------------
# Core actions
# ------------------------------------------------------------------

def execute_pcl(patran: gw.Win32Window, command: str) -> None:
    """Paste a PCL command into Patran's command bar and press Enter."""
    _user32.SetForegroundWindow(patran._hWnd)
    time.sleep(0.2)

    pyautogui.click(patran.left + patran.width // 2,
                    patran.top + patran.height - 40)
    time.sleep(0.1)

    pyperclip.copy(command)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyautogui.press("enter")


def handle_popups(timeout: float, idle_limit: float) -> int:
    """Click Yes on every 'Message' popup until idle for *idle_limit* seconds.

    Returns the number of popups handled.
    """
    handled = 0
    deadline = time.time() + timeout
    last_popup_time = time.time()

    while time.time() < deadline:
        popup = find_message_popup()
        if popup is not None:
            _user32.SetForegroundWindow(popup._hWnd)
            time.sleep(0.08)
            pyautogui.hotkey("alt", "y")
            pyautogui.press("enter")
            handled += 1
            last_popup_time = time.time()
            if handled % 10 == 0:
                print(f"  ... {handled} popups handled")
            time.sleep(0.05)
        else:
            if handled > 0 and (time.time() - last_popup_time) > idle_limit:
                break
            time.sleep(0.05)

    return handled


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Patran .ses.01 session file via the PCL command bar."
    )
    parser.add_argument("ses_file",
                        help="Path to the .ses.01 file to play.")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Max seconds to wait for popups (default: 300).")
    parser.add_argument("--idle", type=float, default=10,
                        help="Stop after this many seconds without popups (default: 10).")
    args = parser.parse_args()

    ses_path = os.path.abspath(args.ses_file)
    if not os.path.isfile(ses_path):
        print(f"Error: file not found: {ses_path}")
        sys.exit(1)

    # 1. Find Patran
    patran = find_patran_window()
    if patran is None:
        print("Error: Patran window not found. Is Patran running?")
        sys.exit(1)
    print(f"Patran: '{patran.title}'")

    # 2. Execute !!input
    pcl_path = ses_path.replace("\\", "/")
    pcl_cmd = f'!!input "{pcl_path}"'
    print(f"Running: {pcl_cmd}")
    execute_pcl(patran, pcl_cmd)

    # 3. Handle popups (exits early after --idle seconds of no popups)
    t0 = time.time()
    n = handle_popups(args.timeout, args.idle)
    elapsed = time.time() - t0
    print(f"Done: {n} popup(s) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
