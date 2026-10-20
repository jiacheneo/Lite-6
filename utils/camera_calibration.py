#!/usr/bin/env python3
"""
camera_calibration.py - intrinsic calibration for the Arm Lab RealSense camera.

Shows a live view of the color stream. Hold a checkerboard in front of the
camera and press SPACE to keep a frame. Frames where the whole board is not
visible are rejected, so every frame you keep is usable. When you have a good
spread of views, click "Calibrate".

Both sets of numbers are printed to the terminal: the camera's own factory
intrinsics, and the ones computed from your frames. The computed result is
also saved to an .npz file next to this script.

Run it inside the lab environment:

    conda activate env550lab
    python camera_calibration.py

This uses PyQt5 for display rather than cv2.imshow, because the lab pins
opencv-python-headless, which has no GUI support at all.
"""

import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QVBoxLayout, QWidget)

# ---------------------------------------------------------------------------
# Checkerboard - EDIT THESE to match your printed target
# ---------------------------------------------------------------------------
BOARD_COLS = 6        # inner corners across (a 10-square-wide board has 9)  # revise to 6
BOARD_ROWS = 7        # inner corners down   (a  7-square-tall board has 6)  # revise to 7
SQUARE_MM = 40.0      # measure one printed square with calipers # revise to 40.0

# ---------------------------------------------------------------------------
# Camera stream - match what the control station uses
# ---------------------------------------------------------------------------
COLOR_W = 1280
COLOR_H = 720
FPS = 30

# ---------------------------------------------------------------------------
# Capture rules
# ---------------------------------------------------------------------------
MIN_FRAMES = 10                       # Calibrate stays disabled below this
GOOD_RMS_PX = 0.5                     # at or below this, the fit is good
OUTPUT_FILE = "camera_intrinsics.npz"

PATTERN = (BOARD_COLS, BOARD_ROWS)


# ---------------------------------------------------------------------------
# Calibration maths (no GUI in here, so it can be tested on its own)
# ---------------------------------------------------------------------------

def board_object_points():
    """3D coordinates of the inner corners in the board's own frame, z = 0."""
    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    return objp * SQUARE_MM


def find_board_fast(gray):
    """Cheap check used for the live preview. Returns (found, corners)."""
    found, corners = cv2.findChessboardCorners(
        gray, PATTERN, flags=cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    return bool(found), corners


def find_board_accurate(gray):
    """Sub-pixel detector used when a frame is actually captured."""
    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    return bool(found), corners


def run_calibration(image_points, image_size):
    """
    Calibrate from a list of corner sets.

    image_points: list of corner arrays, one per captured frame
    image_size:   (width, height)

    Returns a dict with the RMS error, K, the distortion coefficients and the
    per-frame reprojection error.
    """
    objp = board_object_points()
    object_points = [objp] * len(image_points)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    per_frame = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(
            image_points[i].reshape(-1, 2) - projected.reshape(-1, 2), axis=1
        ).mean()
        per_frame.append(float(err))

    return {"rms": float(rms), "K": K, "dist": dist,
            "per_frame": per_frame, "image_size": image_size}


def format_report(result, factory):
    """Build the terminal report comparing the factory and computed intrinsics."""
    K, dist = result["K"], result["dist"]
    w, h = result["image_size"]
    lines = []
    lines.append("")
    lines.append("=" * 62)
    lines.append("  factory intrinsics (from the camera)")
    lines.append("=" * 62)
    if factory is None:
        lines.append("  unavailable")
    else:
        lines.append(f"  resolution : {factory['width']} x {factory['height']}")
        lines.append(f"  fx, fy     : {factory['fx']:.2f}, {factory['fy']:.2f}")
        lines.append(f"  cx, cy     : {factory['ppx']:.2f}, {factory['ppy']:.2f}")
        lines.append(f"  model      : {factory['model']}")
        lines.append("  coeffs     : "
                     + ", ".join(f"{c:.5f}" for c in factory["coeffs"]))

    lines.append("")
    lines.append("=" * 62)
    lines.append(f"  calculated intrinsics (from {len(result['per_frame'])} frames)")
    lines.append("=" * 62)
    lines.append(f"  resolution : {w} x {h}")
    lines.append(f"  fx, fy     : {K[0, 0]:.2f}, {K[1, 1]:.2f}")
    lines.append(f"  cx, cy     : {K[0, 2]:.2f}, {K[1, 2]:.2f}")
    lines.append("  coeffs     : "
                 + ", ".join(f"{c:.5f}" for c in dist.ravel()))
    lines.append(f"  RMS reprojection error : {result['rms']:.3f} px")

    if factory is not None:
        lines.append("")
        lines.append("  difference from factory:")
        lines.append(f"    fx {K[0, 0] - factory['fx']:+.2f} px    "
                     f"fy {K[1, 1] - factory['fy']:+.2f} px")
        lines.append(f"    cx {K[0, 2] - factory['ppx']:+.2f} px    "
                     f"cy {K[1, 2] - factory['ppy']:+.2f} px")

    lines.append("")
    lines.append("  per-frame reprojection error:")
    for i, err in enumerate(result["per_frame"]):
        flag = "   <-- high, consider recapturing" if err > 1.0 else ""
        lines.append(f"    frame {i:2d} : {err:.3f} px{flag}")

    lines.append("")
    if result["rms"] <= GOOD_RMS_PX:
        lines.append(f"  RMS is at or below {GOOD_RMS_PX} px - this is a good fit.")
    elif result["rms"] <= 1.0:
        lines.append("  RMS is acceptable but not great. More tilted views, and "
                     "views near the edges of the frame, will improve it.")
    else:
        lines.append("  RMS is above 1.0 px. Check that BOARD_COLS/BOARD_ROWS count "
                     "INNER CORNERS and that SQUARE_MM matches your printed board.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CalibrationWindow(QMainWindow):

    def __init__(self, pipeline, factory):
        super().__init__()
        self.pipeline = pipeline
        self.factory = factory
        self.captures = []        # corner arrays, one per accepted frame
        self.outlines = []        # board outlines, for the coverage overlay
        self.frame = None         # latest BGR frame from the camera
        self.preview_ok = False

        self.setWindowTitle("Arm Lab - camera calibration")

        self.video = QLabel("starting camera...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(960, 540)
        self.video.setStyleSheet("background:#202020; color:#dddddd;")

        self.status = QLabel("Hold the checkerboard in view.")
        self.status.setStyleSheet("font-size:15px; padding:4px;")
        self.count = QLabel("")
        self.count.setStyleSheet("font-size:15px; padding:4px; color:#444444;")
        self._update_count()

        self.btn_capture = QPushButton("Capture  (Space)")
        self.btn_undo = QPushButton("Undo last")
        self.btn_calib = QPushButton("Calibrate")
        self.btn_quit = QPushButton("Quit")
        for b in (self.btn_capture, self.btn_undo, self.btn_calib, self.btn_quit):
            b.setFocusPolicy(Qt.NoFocus)   # so Space always reaches the window
            b.setMinimumHeight(34)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_calib.clicked.connect(self.calibrate)
        self.btn_quit.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_capture)
        buttons.addWidget(self.btn_undo)
        buttons.addWidget(self.btn_calib)
        buttons.addStretch()
        buttons.addWidget(self.btn_quit)

        self._update_count()          # now that the buttons exist, set their state

        layout = QVBoxLayout()
        layout.addWidget(self.video, stretch=1)
        layout.addWidget(self.status)
        layout.addWidget(self.count)
        layout.addLayout(buttons)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / FPS))

    # -- capture state ------------------------------------------------------

    def _update_count(self):
        n = len(self.captures)
        if n < MIN_FRAMES:
            self.count.setText(f"Captured {n} - need at least {MIN_FRAMES}. "
                               f"Tilt the board and cover the edges of the frame.")
        else:
            self.count.setText(f"Captured {n}. More varied views still help; "
                               f"click Calibrate when you are ready.")
        if hasattr(self, "btn_calib"):
            self.btn_calib.setEnabled(n >= MIN_FRAMES)
            self.btn_undo.setEnabled(n > 0)

    def _say(self, text, color="#000000"):
        self.status.setText(text)
        self.status.setStyleSheet(f"font-size:15px; padding:4px; color:{color};")

    # -- frame loop ---------------------------------------------------------

    def tick(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return
        color = frames.get_color_frame()
        if not color:
            return

        bgr = np.asanyarray(color.get_data())
        self.frame = bgr

        # Cheap detection on a half-size image, just for the live indicator.
        small = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.preview_ok, corners = find_board_fast(gray)

        view = bgr.copy()
        for quad in self.outlines:                       # coverage so far
            cv2.polylines(view, [quad.astype(np.int32)], True, (90, 90, 90), 1)
        if self.preview_ok and corners is not None:
            cv2.drawChessboardCorners(view, PATTERN, corners * 2.0, True)

        if self.preview_ok:
            self._say("Board detected - press SPACE to capture.", "#146c2b")
        else:
            self._say("No board visible. Show the whole board to the camera.", "#a03030")

        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # -- actions ------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def capture(self):
        if self.frame is None:
            return
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_board_accurate(gray)
        if not found:
            self._say("Not captured - the whole board was not visible. "
                      "Move back or straighten it slightly.", "#a03030")
            return

        self.captures.append(corners)
        pts = corners.reshape(-1, 2)
        self.outlines.append(np.array([
            pts[0], pts[BOARD_COLS - 1],
            pts[BOARD_COLS * BOARD_ROWS - 1], pts[BOARD_COLS * (BOARD_ROWS - 1)],
        ]))
        self._update_count()
        self._say(f"Captured frame {len(self.captures)}.", "#146c2b")

    def undo(self):
        if self.captures:
            self.captures.pop()
            self.outlines.pop()
            self._update_count()
            self._say("Removed the last capture.")

    def calibrate(self):
        n = len(self.captures)
        if n < MIN_FRAMES:
            return
        self._say(f"Calibrating from {n} frames...")
        QApplication.processEvents()

        result = run_calibration(self.captures, (COLOR_W, COLOR_H))
        print(format_report(result, self.factory))

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
        np.savez(out,
                 K=result["K"], dist=result["dist"],
                 image_size=np.array(result["image_size"]),
                 rms=result["rms"], frames=n)
        print(f"  saved to {out}\n")

        K = result["K"]
        self._say(f"Done - RMS {result['rms']:.3f} px, "
                  f"fx {K[0, 0]:.1f}, fy {K[1, 1]:.1f}, "
                  f"cx {K[0, 2]:.1f}, cy {K[1, 2]:.1f}. "
                  f"Full report in the terminal.",
                  "#146c2b" if result["rms"] <= 1.0 else "#a05a00")

    def closeEvent(self, event):
        self.timer.stop()
        self.pipeline.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------

def start_camera():
    """Start the color stream. Returns (pipeline, factory_intrinsics_dict)."""
    if len(rs.context().query_devices()) == 0:
        print("No RealSense camera found.")
        print("  - check the USB cable, and that it is a USB 3.x port")
        print("  - run 'rs-enumerate-devices -s' to confirm the camera is seen")
        print("  - close realsense-viewer if it is open; it holds the camera")
        return None, None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    factory = {"width": intr.width, "height": intr.height,
               "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
               "model": str(intr.model), "coeffs": list(intr.coeffs)}
    return pipeline, factory


def main():
    print(__doc__)
    print(f"Checkerboard: {BOARD_COLS} x {BOARD_ROWS} inner corners, "
          f"{SQUARE_MM} mm squares.")
    print("If the board is never detected, that count is the first thing to "
          "check - it is INNER CORNERS, not squares.\n")

    pipeline, factory = start_camera()
    if pipeline is None:
        return 1

    app = QApplication(sys.argv)
    window = CalibrationWindow(pipeline, factory)
    window.resize(1040, 760)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
