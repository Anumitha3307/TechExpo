#!/usr/bin/env python3
"""
Optimized color.py — HSV color range tuner for SteeringWheel project.
Use this to find lower/upper HSV bounds for your marker color.

Press 's' to print current HSV range to console.
Press 'q' to quit.
"""

import cv2
import numpy as np
from imutils.video import VideoStream
import time

def nothing(x):
    pass

def create_trackbars(window="Trackbars"):
    cv2.namedWindow(window)
    cv2.createTrackbar("H Lower", window, 0, 180, nothing)
    cv2.createTrackbar("S Lower", window, 0, 255, nothing)
    cv2.createTrackbar("V Lower", window, 0, 255, nothing)
    cv2.createTrackbar("H Upper", window, 180, 180, nothing)
    cv2.createTrackbar("S Upper", window, 255, 255, nothing)
    cv2.createTrackbar("V Upper", window, 255, 255, nothing)

def get_trackbar_values(window="Trackbars"):
    hL = cv2.getTrackbarPos("H Lower", window)
    sL = cv2.getTrackbarPos("S Lower", window)
    vL = cv2.getTrackbarPos("V Lower", window)
    hU = cv2.getTrackbarPos("H Upper", window)
    sU = cv2.getTrackbarPos("S Upper", window)
    vU = cv2.getTrackbarPos("V Upper", window)
    return np.array([hL, sL, vL]), np.array([hU, sU, vU])

def main():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    create_trackbars()

    print("Adjust sliders until only your marker is visible in the mask window.")
    print("Press 's' to show HSV range, 'q' to quit.")

    while True:
        frame = vs.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = get_trackbar_values()
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Filtered", res)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Lower HSV: {lower.tolist()}, Upper HSV: {upper.tolist()}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
