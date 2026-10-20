"""Task 2.8c: move to a fixed waypoint, record on click, then advance."""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from fixture_measurement import (LOCATIONS, WAYPOINTS, REGIONS,
                                 TAG_OFFSET, TCP_RPY_DEG, pose_matrix, tag_truth,
                                 measurement_route, export_report)


class FixtureMeasurementDialog(QDialog):
    def __init__(self, arm, camera, parent=None):
        super().__init__(parent)
        self.arm, self.camera, self.sdk = arm, camera, arm.xarm
        self.phase = 'idle'
        self.rows = []
        self.setWindowTitle('Task 2.8c — Record waypoints')
        self.resize(600, 220)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            'Tag 6 · flange offset (25, 0, 75) mm\n'
            f'TCP roll / pitch / yaw: {TCP_RPY_DEG} degrees\n'
            'First waypoint → attach fixture → Record & next at each point.'))
        self.status = QLabel('Go to the first waypoint, then attach the tag there.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.action = QPushButton('Go to first point')
        self.action.clicked.connect(self.advance)
        self.action.setAutoDefault(False)
        buttons.addWidget(self.action)
        self.grip_buttons = []
        for label, method, state in [('Open gripper', arm.open_gripper, 'open'),
                                     ('Close gripper', arm.close_gripper, 'closed')]:
            button = QPushButton(label)
            button.setAutoDefault(False)
            button.setEnabled(False)
            button.clicked.connect(lambda checked=False, fn=method, value=state:
                                   self.actuate_gripper(fn, value))
            buttons.addWidget(button)
            self.grip_buttons.append(button)
        stop = QPushButton('STOP')
        stop.setAutoDefault(False)
        stop.clicked.connect(lambda: self.abort('Stopped by operator.'))
        buttons.addWidget(stop)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def advance(self):
        try:
            if self.phase == 'idle':
                self.start_measurement()
            elif self.phase == 'ready':
                self.record()
        except Exception as exc:
            self.abort(str(exc))

    def checked(self, code, operation):
        if code != 0:
            raise RuntimeError(f'{operation}: SDK code {code}')

    def pose(self):
        code, pose = self.sdk.get_position(is_radian=True)
        self.checked(code, 'Read TCP pose')
        pose_matrix(pose)
        return np.asarray(pose, dtype=float)

    def check_arm(self):
        if not self.sdk.connected or self.arm.has_fault() or self.sdk.state in (3, 4):
            raise RuntimeError('Arm disconnected, paused, or faulted.')

    def start_measurement(self):
        self.check_arm()
        if not self.camera.extrinsics_from_tags:
            raise RuntimeError('Calibrate from board tags before this test.')
        self.checked(self.sdk.set_mode(0), 'Position mode')
        self.checked(self.sdk.set_state(0), 'Enable motion')
        self.tcp_offset = np.asarray(self.sdk.tcp_offset, dtype=float).copy()
        pose_matrix(self.tcp_offset)
        self.route = measurement_route(self.pose())
        bias = getattr(self.camera, 'active_depth_bias', lambda: None)()
        self.metadata = {
            'depth_bias_correction': bias.to_dict() if bias is not None else None,
            'option': 1, 'tag_id': 6, 'tag_size_mm': 25,
            'flange_tag_offset_mm': TAG_OFFSET,
            'offset_note': 'Approximate user measurement; contributes to truth uncertainty.',
            'sdk_tcp_offset': self.tcp_offset.tolist(),
            'intrinsics': self.camera.intrinsic_matrix.tolist(),
            'world_to_camera': self.camera.extrinsic_matrix.tolist(),
            'depth_scale_mm': self.camera.depth_scale_mm,
            'locations': LOCATIONS, 'regions': REGIONS, 'waypoints': WAYPOINTS,
            'tcp_target_rpy_deg': TCP_RPY_DEG,
            'height_grouping': 'Commanded TCP height; actual tag truth Z saved per row.',
            'samples_per_point': 1,
        }
        self.folder = Path(__file__).resolve().parents[2] / 'fixture_results_2.8c' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        export_report(self.folder, self.rows, self.metadata)
        if bias is not None:
            np.save(self.folder / 'depth_bias_mm.npy', bias.bias_mm)
        for button in self.grip_buttons:
            button.setEnabled(False)
        self.route_index = -1
        self.next_move()

    def actuate_gripper(self, method, state):
        if self.phase == 'ready' and not self.rows:
            method()
            self.ready_at = time.monotonic()  # Discard images taken before adjusting the fixture.
            if self.parent() is not None:
                self.parent().sm.gripper_state = state

    def wait_for_motion(self, phase, timeout=90):
        self.phase, self.deadline = phase, time.monotonic() + timeout
        self.arrived_since = None
        self.action.setEnabled(False)

    def next_move(self):
        for button in self.grip_buttons:
            button.setEnabled(False)
        self.route_index += 1
        if self.route_index == len(self.route):
            self.phase = 'done'
            self.action.setEnabled(False)
            self.status.setText(f'Complete: {len(self.rows)} points. Arm stays here.\n{self.folder}')
            return
        self.target, self.location_index = self.route[self.route_index]
        self.checked(self.sdk.set_position(
            **dict(zip(('x', 'y', 'z', 'roll', 'pitch', 'yaw'), self.target)),
            speed=30, mvacc=100, is_radian=True, wait=False), 'Cartesian move')
        self.wait_for_motion('moving')
        self.status.setText(f'Moving TCP to {self.target[:3]} mm. {len(self.rows)}/{len(WAYPOINTS)} recorded.')

    def stable(self, arrived, now):
        if not arrived or self.sdk.get_is_moving():
            self.arrived_since = None
            return False
        if self.arrived_since is None:
            self.arrived_since = now
        return now - self.arrived_since >= 1.5

    def at_target(self, pose):
        position_error = np.linalg.norm(pose[:3] - self.target[:3])
        rotation_error = (Rotation.from_euler('xyz', pose[3:]).inv()
                          * Rotation.from_euler('xyz', self.target[3:])).magnitude()
        return position_error < 2 and rotation_error < np.deg2rad(1)

    def tick(self):
        if self.phase in ('idle', 'done', 'stopped'):
            return
        try:
            self.check_arm()
            if self.phase == 'ready':
                return  # No timeout or automatic recording while waiting for the operator.
            now = time.monotonic()
            if now > self.deadline:
                raise RuntimeError('Motion timeout.')
            if self.stable(self.at_target(self.pose()), now):
                if self.location_index is None:
                    self.next_move()
                else:
                    self.phase, self.ready_at = 'ready', now
                    self.action.setText('Record & next' if len(self.rows) + 1 < len(WAYPOINTS) else 'Record & finish')
                    self.action.setEnabled(True)
                    self.status.setText(f'Point {len(self.rows)+1}/{len(WAYPOINTS)}: TCP {self.target[:3]} mm. Press Record when ready.')
                    if not self.rows:
                        for button in self.grip_buttons:
                            button.setEnabled(True)
                        self.status.setText('At the first waypoint. Attach the tag, then press Record & next to start.')
        except Exception as exc:
            self.abort(str(exc))

    def record(self):
        self.check_arm()
        if not np.allclose(self.sdk.tcp_offset, self.tcp_offset, atol=1e-6, rtol=0):
            raise RuntimeError('TCP offset changed during measurement.')
        pose = self.pose()
        if self.sdk.get_is_moving() or not self.at_target(pose):
            raise RuntimeError('Arm moved away from the measurement waypoint.')
        snapshot = self.camera.tag_measurement_snapshot
        if snapshot is None or snapshot[0] <= self.ready_at or time.monotonic() - snapshot[0] > 1:
            self.status.setText('No fresh camera frame. Stay here and press Record again.')
            return
        stamp, rgb, depth, detections = snapshot
        tags = [tag for tag in detections if tag.tag_id == 6]
        if len(tags) != 1:
            self.status.setText('Tag 6 is not uniquely visible. Adjust visibility and press Record again.')
            return
        u, v = np.rint(tags[0].center).astype(int)
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            self.status.setText('Tag center outside depth image. Press Record again.')
            return
        estimate = self.camera.camera_to_world(
            self.camera.depth_to_camera_point(u, v, depth[v, u]))
        if estimate is None:
            self.status.setText('No valid tag depth. Stay here and press Record again.')
            return
        truth = tag_truth(pose, self.tcp_offset)
        error = estimate - truth
        row = {'point': len(self.rows)+1, 'location': self.location_index+1,
               'region': REGIONS[self.location_index],
               **{f'tcp_target_{axis}_mm': value for axis, value in zip('xyz', self.target[:3])}}
        for name, values in [('truth', truth), ('estimate', estimate), ('error', error)]:
            row.update({f'{name}_{axis}_mm': float(value) for axis, value in zip('xyz', values)})
        row['distance_error_mm'] = float(np.linalg.norm(error))
        np.savez_compressed(self.folder / f'point_{len(self.rows)+1:02d}.npz',
                            truth=truth, estimate=estimate, tcp_pose=pose,
                            pixel_depth=[u, v, float(depth[v, u])],
                            frame_monotonic=stamp, rgb=rgb, depth_raw=depth)
        # Only advance once both the snapshot and report have been saved.
        export_report(self.folder, self.rows + [row], self.metadata)
        self.rows.append(row)
        self.next_move()

    def abort(self, message):
        self.arm.stop()
        self.phase = 'stopped'
        self.action.setEnabled(False)
        for button in self.grip_buttons:
            button.setEnabled(False)
        self.status.setText(message + '\nMotion stopped. Close dialog; use Go Home to recover.')

    def done(self, result):
        self.timer.stop()
        if self.phase == 'moving':
            self.arm.stop()
        super().done(result)
