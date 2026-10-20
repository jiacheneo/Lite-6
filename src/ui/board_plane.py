"""Select empty-board pixels on a frozen frame and export Task 2.8b."""
from copy import copy
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
)

from board_plane import export_board_report


class SelectionImage(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())


class BoardPlaneDialog(QDialog):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Board Z measurement')
        # Freeze arrays and calibration without starting a second camera.
        self.camera = copy(camera)
        for field in ('video_frame', 'depth_frame_raw', 'intrinsic_matrix',
                      'extrinsic_matrix', 'extrinsic_matrix_inv'):
            setattr(self.camera, field, getattr(camera, field).copy())
        self.polygon = []
        self.exclusions = []
        self.pending = []

        layout = QVBoxLayout(self)
        instructions = QLabel(
            'Frozen RGB frame. Clear the board and calibrate before opening this window.\n'
            'Click around the board in boundary order, then Use as board. '
            'Circle any arm, blocks or rails and press Exclude region.\n'
            'Save report measures world Z against zero using the captured depth and calibration. '
            'Reopen this window for an independent measurement.'
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.image = SelectionImage()
        h, w = self.camera.video_frame.shape[:2]
        scale = min(960/w, 540/h, 1)
        self.image.setFixedSize(round(w*scale), round(h*scale))
        self.image.clicked.connect(self._add_point)
        layout.addWidget(self.image)

        row = QHBoxLayout()
        for title, callback in (
            ('Use as board', self._set_board), ('Exclude region', self._exclude),
            ('Undo point', self._undo), ('Reset', self._reset),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel('Sample spacing (pixels):'))
        self.step = QSpinBox()
        self.step.setRange(5, 100)
        self.step.setValue(20)
        export_row.addWidget(self.step)
        save = QPushButton('Save report')
        save.clicked.connect(self._save)
        export_row.addWidget(save)
        close = QPushButton('Close')
        close.clicked.connect(self.accept)
        export_row.addWidget(close)
        layout.addLayout(export_row)
        self.status = QLabel('Select the board boundary (at least three points).')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self._redraw()

    def _add_point(self, x, y):
        h, w = self.camera.video_frame.shape[:2]
        self.pending.append((min(w-1, int(x*w/self.image.width())),
                             min(h-1, int(y*h/self.image.height()))))
        self._redraw()

    def _finish_polygon(self):
        if len(self.pending) < 3 or cv2.contourArea(np.asarray(self.pending, np.int32)) < 1:
            self.status.setText('Select at least three points around a nonzero area.')
            return None
        polygon, self.pending = self.pending, []
        return polygon

    def _set_board(self):
        polygon = self._finish_polygon()
        if polygon is not None:
            self.polygon = polygon
            self.status.setText('Board selected. Exclude any objects, then Save report.')
            self._redraw()

    def _exclude(self):
        polygon = self._finish_polygon()
        if polygon is not None:
            self.exclusions.append(polygon)
            self.status.setText(f'{len(self.exclusions)} excluded region(s).')
            self._redraw()

    def _undo(self):
        if self.pending:
            self.pending.pop()
        self._redraw()

    def _reset(self):
        self.polygon, self.exclusions, self.pending = [], [], []
        self.status.setText('Selection cleared. Select the board boundary.')
        self._redraw()

    def _redraw(self):
        frame = self.camera.video_frame.copy()
        for polygon, color in [(self.polygon, (0, 255, 0))] + [
            (p, (255, 255, 0)) for p in self.exclusions
        ]:
            if polygon:
                cv2.polylines(frame, [np.asarray(polygon, np.int32)], True, color, 2)
        if self.pending:
            cv2.polylines(frame, [np.asarray(self.pending, np.int32)], False, (255, 255, 255), 2)
            for point in self.pending:
                cv2.circle(frame, point, 4, (255, 255, 255), -1)
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(self.image.size()))

    def _save(self):
        if not self.polygon or self.pending:
            self.status.setText('Finish the board/exclusion selection before saving.')
            return
        root = Path(__file__).resolve().parents[2] / 'board_results_2.8b'
        try:
            output, stats = export_board_report(
                self.camera, self.polygon, self.exclusions, root, self.step.value(),
            )
        except (ValueError, OSError, cv2.error) as exc:
            self.status.setText(f'Could not save report: {exc}')
            return
        self.status.setText(
            f"Saved {stats['valid_count']} valid samples. "
            f"Mean |Z|: {stats['mean_absolute_z_mm']:.2f} mm; "
            f"max |Z|: {stats['max_absolute_z_mm']:.2f} mm.\n{output}"
        )
