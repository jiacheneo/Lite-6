"""Task 3.5: open, place a block, record, export."""
import time
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QInputDialog,
                             QLabel, QPushButton, QVBoxLayout)

from detector_measurement import MeasurementSession, export_report, trial_plan


class DetectorMeasurementDialog(QDialog):
    def __init__(self, camera, parent=None, output_root=None):
        super().__init__(parent)
        self.camera = camera
        self.output_root = Path(output_root) if output_root is not None else (
            Path(__file__).resolve().parents[2] / 'detector_results')
        self.session = None  # Created automatically on the first record; no setup step.
        self.last_recorded_at = time.monotonic()
        self.camera.detector_measurement_snapshot = None
        self.camera.detector_measurement_enabled = True
        self.finished.connect(self.cleanup)
        self.setWindowTitle('Task 3.5 — Detector measurements')
        self.resize(800, 590)
        layout = QVBoxLayout(self)
        instructions = QLabel(
            'Place ONE block at the prompted coordinates. Remove your hand, then Record & next.\n'
            'Lift and replace it each time. Order: 25 mm → 38 mm; red → orange → yellow → green → blue → purple.')
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet('font-size: 17px; font-weight: bold;')
        layout.addWidget(self.prompt)
        self.preview = QLabel('Waiting for the camera…')
        self.preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview)
        row = QHBoxLayout()
        for label, method in [
                ('Record & next', self.record), ('Record miss', self.record_miss),
                ('Undo last trial', self.undo), ('Export report', self.export)]:
            button = QPushButton(label)
            button.setAutoDefault(False)
            button.clicked.connect(lambda checked=False, fn=method: self.run_action(fn))
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel('Ready. Results save automatically after each record.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_prompt()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(150)

    @property
    def next_trial(self):
        return self.session.next_trial if self.session is not None else trial_plan()[0]

    def run_action(self, method):
        try:
            method()
        except Exception as exc:
            self.update_prompt()
            self.status.setText(str(exc))

    def cleanup(self, result=0):
        self.timer.stop()
        self.camera.detector_measurement_enabled = False
        self.camera.detector_measurement_snapshot = None

    def update_prompt(self):
        trial = self.next_trial
        if trial is None:
            self.prompt.setText('Complete: 180/180. Export report, or Undo last trial to correct the final measurement.')
        else:
            self.prompt.setText(
                f'Trial {trial["trial_id"]}/180 · {trial["point"]} ({trial["truth_x_mm"]}, {trial["truth_y_mm"]}) mm\n'
                f'{trial["nominal_size_mm"]} mm {trial["truth_color"].upper()} · placement {trial["repetition"]}/3')

    def show_snapshot(self, snapshot, numbered=False):
        preview = snapshot['preview']
        if numbered:
            preview = preview.copy()
            for i, d in enumerate(snapshot['detections']):
                x, y = (round(v) for v in d['center_px'])
                cv2.putText(preview, f'#{i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 3)
                cv2.putText(preview, f'#{i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 1)
        h, w = preview.shape[:2]
        image = QImage(preview.data, w, h, preview.strides[0], QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(760, 390, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_preview(self):
        snapshot = self.camera.detector_measurement_snapshot
        if not self.camera.camera_connected or snapshot is None or time.monotonic() - snapshot['captured_at'] > 2:
            self.preview.setText('Waiting for a fresh calibrated camera frame…')
            return
        self.show_snapshot(snapshot)

    def record(self, missed=False):
        if self.next_trial is None:
            raise ValueError('All 180 trials are recorded.')
        if time.monotonic() - self.last_recorded_at < 1:
            raise ValueError('Lift and replace the block before recording the next placement.')
        snapshot = self.camera.detector_measurement_snapshot
        if (not self.camera.camera_connected or snapshot is None
                or time.monotonic() - snapshot['captured_at'] > 2):
            raise ValueError('No fresh calibrated frame. Check the camera, Calibrate and Flatten Table.')
        if snapshot['captured_at'] <= self.last_recorded_at:
            raise ValueError('Wait for a new frame after replacing the block.')
        detections = snapshot['detections']
        selected = None if missed or not detections else 0
        if not missed and len(detections) > 1:
            # Only ambiguous frames need a selection; keep the displayed frame fixed.
            self.timer.stop()
            try:
                self.show_snapshot(snapshot, numbered=True)
                options = [f'#{i+1}: {d.get("color", "unknown")} {d.get("size", "?")}'
                           for i, d in enumerate(detections)]
                options.append('Missed — none of these is the tested block')
                choice, accepted = QInputDialog.getItem(
                    self, 'Select tested block', 'Match the block to its number in the image:',
                    options, 0, False)
                if not accepted:
                    return
                index = options.index(choice)
                selected = index if index < len(detections) else None
            finally:
                self.timer.start(150)
        if self.session is None:
            self.session = MeasurementSession.create(
                self.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f'), 'Not recorded')
        count_before = len(self.session.rows)
        try:
            self.session.record(snapshot, selected)
        finally:
            # A CSV failure after the JSON commit must not count this frame twice.
            if len(self.session.rows) > count_before:
                self.last_recorded_at = time.monotonic()
        self.update_prompt()
        self.status.setText(f'Saved {len(self.session.rows)}/180. Lift and replace the block.\n{self.session.folder}')
        if self.next_trial is None:
            self.export()

    def record_miss(self):
        self.record(missed=True)

    def undo(self):
        if self.session is None or not self.session.rows:
            raise ValueError('No recorded trial to undo.')
        self.session.undo()
        self.last_recorded_at = time.monotonic()
        self.update_prompt()
        self.status.setText('Last measurement removed. Place the prompted block and record again.')

    def export(self):
        if self.session is None:
            raise ValueError('Record at least one trial first.')
        output = export_report(self.session)
        self.status.setText(f'Report saved: {output}')
