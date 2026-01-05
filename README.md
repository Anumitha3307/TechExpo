# TechExpo – Vision-Based Racing Game Controller

This project is a Python-based racing game controller that uses computer vision
to control steering and acceleration through real-time camera input.

## How It Works
- A colored marker is detected using HSV color thresholding (OpenCV)
- The position of the marker determines steering direction:
  - Left → Press `A`
  - Right → Press `D`
  - Bottom region → Press `Space` (Nitro)
- Keyboard inputs are triggered programmatically to control the game

## Files Overview
- `color.py` – HSV color range tuner using trackbars
- `directkeys.py` – Handles keyboard key press and release
- `steering.py` – Core logic for marker detection and steering decisions
- `tutorial.py` – Combined demo for visualization and testing

## Technologies Used
- Python
- OpenCV
- NumPy
- imutils

## Purpose
This project was developed as a learning exercise to understand:
- Computer vision basics
- Real-time input handling
- Game automation and control logic
