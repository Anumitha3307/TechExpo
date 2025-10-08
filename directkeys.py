#!/usr/bin/env python3
"""
Optimized directkeys.py
Provides PressKey and ReleaseKey for steering actions.

Windows: uses ctypes to call keybd_event.
Others: falls back to pynput (requires pip install pynput).
"""

import platform
import time

# Key mappings (virtual key codes for A, D, SPACE)
A, D, Space = 0x1E, 0x20, 0x39  # default values for Windows

if platform.system() == "Windows":
    import ctypes
    SendInput = ctypes.windll.user32.keybd_event

    def PressKey(hexKeyCode):
        SendInput(hexKeyCode, 0, 0, 0)

    def ReleaseKey(hexKeyCode):
        SendInput(hexKeyCode, 0, 0x0002, 0)

else:
    # fallback for Linux/macOS
    try:
        from pynput.keyboard import Controller, Key
        keyboard = Controller()

        _map = {A: 'a', D: 'd', Space: Key.space}

        def PressKey(hexKeyCode):
            key = _map.get(hexKeyCode)
            if key:
                keyboard.press(key)

        def ReleaseKey(hexKeyCode):
            key = _map.get(hexKeyCode)
            if key:
                keyboard.release(key)

    except ImportError:
        print("[Warning] pynput not found — key presses disabled.")
        def PressKey(k): pass
        def ReleaseKey(k): pass

if __name__ == "__main__":
    print("Testing key presses...")
    PressKey(A); time.sleep(0.2); ReleaseKey(A)
    PressKey(D); time.sleep(0.2); ReleaseKey(D)
    PressKey(Space); time.sleep(0.2); ReleaseKey(Space)
    print("Done.")
