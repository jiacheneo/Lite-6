"""Four table samples on a frozen top-down depth view."""
from copy import copy

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ui.board_plane import SelectionImage


class TablePlaneDialog(QDialog):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Flatten table: four corners')
        self.camera = camera
        self.homography = camera.workspace_homography
        w, h = camera.workspace_output_size()
        self.depth = cv2.warpPerspective(
            camera.depth_frame_raw.copy(), self.homography, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        preview = copy(camera)
        preview.colorize_depth_frame(frame=self.depth)
        self.frame = preview.depth_frame_rgb
        self.points = []
        layout = QVBoxLayout(self)
        instructions = QLabel(
            'Click four widely separated table corners on this frozen depth image.\n'
            'Keep each 9x9 window on bare table; avoid tags, rails, blocks and the arm.')
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.image = SelectionImage()
        scale = min(960 / w, 540 / h, 1)
        self.image.setFixedSize(round(w * scale), round(h * scale))
        self.image.clicked.connect(self._add_point)
        layout.addWidget(self.image)
        self.status = QLabel('Selected 0/4 corners.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        for title, callback in [('Reset points', self._reset),
                                ('Apply', self._apply), ('Cancel', self.reject)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self._redraw()

    def _add_point(self, x, y):
        if len(self.points) == 4:
            return
        h, w = self.depth.shape
        self.points.append((int(x * w / self.image.width()),
                            int(y * h / self.image.height())))
        self.status.setText(f'Selected {len(self.points)}/4 corners. Apply when ready.')
        self._redraw()

    def _reset(self):
        self.points = []
        self.status.setText('Selected 0/4 corners.')
        self._redraw()

    def _redraw(self):
        frame = self.frame.copy()
        for index, (u, v) in enumerate(self.points, 1):
            cv2.rectangle(frame, (u-4, v-4), (u+4, v+4), (255, 255, 255), 1)
            cv2.putText(frame, str(index), (u+7, v-7),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(self.image.size()))

    def _apply(self):
        if self.camera.workspace_homography is not self.homography:
            self.status.setText('Calibration changed. Close and reopen this window.')
            return
        try:
            self.camera.fit_workspace_table_plane(self.points, self.depth)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.status.setText(str(exc))
            return
        self.accept()
