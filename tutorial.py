#!/usr/bin/env python3
"""
Optimized tutorial.py — Combined demo that visualizes color detection and steering decisions.

Press:
 - 'q' to quit
 - 's' to print current HSV range
"""

import cv2
import imutils
import numpy as np
import time
from imutils.video import VideoStream

# Use functions from local modules
from directkeys import PressKey, ReleaseKey, A, D, Space

def nothing(x): pass

def create_trackbars():
    cv2.namedWindow("Controls")
    cv2.createTrackbar("H Lower", "Controls", 53, 180, nothing)
    cv2.createTrackbar("S Lower", "Controls", 55, 255, nothing)
    cv2.createTrackbar("V Lower", "Controls", 209, 255, nothing)
    cv2.createTrackbar("H Upper", "Controls", 180, 180, nothing)
    cv2.createTrackbar("S Upper", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V Upper", "Controls", 255, 255, nothing)

def get_hsv_range():
    hL = cv2.getTrackbarPos("H Lower", "Controls")
    sL = cv2.getTrackbarPos("S Lower", "Controls")
    vL = cv2.getTrackbarPos("V Lower", "Controls")
    hU = cv2.getTrackbarPos("H Upper", "Controls")
    sU = cv2.getTrackbarPos("S Upper", "Controls")
    vU = cv2.getTrackbarPos("V Upper", "Controls")
    return np.array([hL, sL, vL]), np.array([hU, sU, vU])

def main():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    create_trackbars()

    print("Adjust HSV sliders until detection works.")
    print("Press 'q' to quit, 's' to print HSV values.")

    while True:
        frame = vs.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=640)

        lower, upper = get_hsv_range()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Filtered", res)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Lower: {lower.tolist()}, Upper: {upper.tolist()}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
