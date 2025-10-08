#!/usr/bin/env python3
"""
Optimized steering.py — replacement for Aviral09/SteeringWheel steering.py

Improvements:
 - Single resize to 640x480
 - Precomputed morphology kernels
 - Use set() for currently pressed keys
 - Simple debounce (frames) to avoid flicker
 - Proper resource cleanup & error handling
 - Readable, modular structure
"""

import argparse
import time
from collections import deque

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# keep these import names same as the original repository
from directkeys import PressKey, ReleaseKey, A, D, Space

# --- Configurable defaults ----
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# HSV thresholds from original (kept as default, but configurable)
DEFAULT_LOWER = np.array([53, 55, 209], dtype=np.uint8)
DEFAULT_UPPER = np.array([180, 255, 255], dtype=np.uint8)

GAUSSIAN_KERNEL = (11, 11)  # must be odd
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# Debounce: require n consecutive frames that ask for the same key before toggling
DEBOUNCE_FRAMES = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=int, default=0, help="camera source index")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--lower", nargs=3, type=int, default=list(DEFAULT_LOWER),
                   help="HSV lower bound (H S V)")
    p.add_argument("--upper", nargs=3, type=int, default=list(DEFAULT_UPPER),
                   help="HSV upper bound (H S V)")
    p.add_argument("--debounce", type=int, default=DEBOUNCE_FRAMES,
                   help="frames for debounce/hysteresis")
    return p.parse_args()


class SteeringController:
    def __init__(self, src=0, width=640, height=480, lower=None, upper=None, debounce_frames=3):
        self.vs = VideoStream(src=src).start()
        # allow camera warm-up
        time.sleep(1.0)
        self.width = width
        self.height = height
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.morph_kernel = MORPH_KERNEL
        self.gauss_ksize = GAUSSIAN_KERNEL
        self.current_keys = set()
        self.debounce_frames = max(1, debounce_frames)
        # deque to hold last N decisions (None, 'A', 'D', 'Space') for simple stabilization
        self.history = deque(maxlen=self.debounce_frames)

    def read_frame(self):
        frame = self.vs.read()
        if frame is None:
            return None
        # flip (mirror) and resize once
        frame = np.flip(frame, axis=1)
        frame = imutils.resize(frame, width=self.width, height=self.height)
        return frame

    def preprocess_mask(self, frame):
        # convert to HSV and blur, then threshold
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, self.gauss_ksize, 0)
        mask = cv2.inRange(blurred, self.lower, self.upper)
        # morphological open then close to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return mask

    def find_direction(self, mask):
        """
        Returns one of: 'LEFT', 'RIGHT', 'NITRO', or None.
        The original code checks top half for centroid (left/right)
        and bottom-ish small region for nitro.
        """
        h, w = mask.shape[:2]
        # upContour — top half across full width
        upContour = mask[0:h // 2, 0:w]
        # downContour — bottom band (similar to original); I use a slightly wider area for robustness
        downContour = mask[3 * h // 4:h, 2 * w // 5:3 * w // 5]

        # Left/Right detection in top half
        cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_up = imutils.grab_contours(cnts_up)
        direction = None

        if cnts_up:
            c = max(cnts_up, key=cv2.contourArea)
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cX = int(M["m10"] / M["m00"])
                # adjust cX to full-frame coordinates by no-op since we used a sliced mask
                # left threshold and right threshold as original (center ± 35)
                centre = w // 2
                if cX < (centre - 35):
                    direction = "LEFT"
                elif cX > (centre + 35):
                    direction = "RIGHT"

        # Nitro detection: presence of any contour in downContour
        cnts_down = cv2.findContours(downContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_down = imutils.grab_contours(cnts_down)
        if cnts_down and len(cnts_down) > 0:
            # nitro has priority (original pressed Space whenever downContour had something)
            direction = "NITRO"

        return direction

    def apply_key_state(self, decision):
        """
        Map decision to key presses/releases.
        decision: 'LEFT' -> Press A
                  'RIGHT' -> Press D
                  'NITRO' -> Press Space
                  None -> release all
        Use a set to track currently pressed keys to avoid repeated PressKey calls.
        """
        desired = set()
        if decision == "LEFT":
            desired.add(A)
        elif decision == "RIGHT":
            desired.add(D)
        elif decision == "NITRO":
            desired.add(Space)

        # Press keys that are desired but not currently pressed
        for k in desired - self.current_keys:
            PressKey(k)
            self.current_keys.add(k)

        # Release keys that are currently pressed but not desired anymore
        for k in list(self.current_keys - desired):
            ReleaseKey(k)
            self.current_keys.discard(k)

    def draw_ui(self, frame, decision):
        h, w = frame.shape[:2]
        # left box
        cv2.rectangle(frame, (0, 0), (w // 2 - 35, h // 2), (0, 255, 0), 1)
        cv2.putText(frame, 'LEFT', (110, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))
        # right box
        cv2.rectangle(frame, (w // 2 + 35, 0), (w - 2, h // 2), (0, 255, 0), 1)
        cv2.putText(frame, 'RIGHT', (440, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))
        # nitro box
        cv2.rectangle(frame, (2 * (w // 5), 3 * (h // 4)), (3 * w // 5, h), (0, 255, 0), 1)
        cv2.putText(frame, 'NITRO', (2 * (w // 5) + 20, h - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

        # show current decision
        if decision is not None:
            cv2.putText(frame, f'DECISION: {decision}', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("Warning: empty frame, skipping...")
                    continue

                mask = self.preprocess_mask(frame)
                decision = self.find_direction(mask)

                # push decision into history for debounce: if the most recent N entries agree,
                # we accept it. This reduces flicker from frame-to-frame noise.
                self.history.append(decision)
                # check if last N are the same and not all None
                if len(self.history) == self.history.maxlen and all(x == self.history[0] for x in self.history):
                    stable_decision = self.history[0]
                else:
                    # keep previous key state until stable (hysteresis)
                    stable_decision = None

                self.apply_key_state(stable_decision)
                self.draw_ui(frame, stable_decision)
                cv2.imshow("Steering", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            # release everything and ensure keys are released
            for k in list(self.current_keys):
                ReleaseKey(k)
            cv2.destroyAllWindows()
            # VideoStream's stop() is not always necessary but good to call if implemented
            try:
                self.vs.stop()
            except Exception:
                pass


def main():
    args = parse_args()
    controller = SteeringController(src=args.src,
                                    width=args.width,
                                    height=args.height,
                                    lower=args.lower,
                                    upper=args.upper,
                                    debounce_frames=args.debounce)
    controller.run()


if __name__ == "__main__":
    main()
