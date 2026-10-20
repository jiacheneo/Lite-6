#!/usr/bin/python
"""Main GUI for the Arm Lab runtime shell."""
import argparse
import sys
from functools import partial

import numpy as np
from PyQt5.QtCore import QEvent, QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

from camera import Camera, VideoThread
from lite6arm import ArmThread, Lite6Arm, JOINT_NAMES
from lite6arm_sim import SimArm, SimArmThread
from state_machine import StateMachine, StateMachineThread
from ui.layout import Ui_MainWindow
from ui.style import STYLE


class Gui(QMainWindow):

    def __init__(self, arm_mode="real", parent=None):
        """Create the GUI, hardware objects, signal wiring, and worker threads."""
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --- Hardware ---
        self.arm_mode = arm_mode  # "sim": MuJoCo sim over LCM; "real": arm hardware
        self.camera = Camera()
        self.arm    = SimArm() if arm_mode == "sim" else Lite6Arm()
        self.sm     = StateMachine(self.arm, self.camera)

        # --- View state ---
        self._current_frames = {"video": None, "depth": None, "tags": None, "workspace": None, "mask": None}
        self._pre_manual_widget_states = {}  # restore panel states after manual mode

        # --- Wire jog buttons (widget pairs defined in layout) ---
        for idx, (btn_neg, btn_pos) in enumerate(self.ui.jog_buttons):
            btn_neg.pressed.connect(partial(self._start_jog, idx, -1))
            btn_neg.released.connect(self._stop_jog)
            btn_pos.pressed.connect(partial(self._start_jog, idx, 1))
            btn_pos.released.connect(self._stop_jog)

        self.ui.videoDisplay.installEventFilter(self)
        QTimer.singleShot(0, self._sync_video_panel_geometry)

        # --- Action buttons ---
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_initial_pose.clicked.connect(self._initial_pose)
        self.ui.chk_manual_mode.stateChanged.connect(self._manual_mode_chk)
        self.ui.btn_open_gripper.clicked.connect(self._open_gripper)
        self.ui.btn_close_gripper.clicked.connect(self._close_gripper)
        self.ui.btn_stop_gripper.clicked.connect(self.arm.stop_gripper)
        self.ui.btn_sleep_arm.clicked.connect(self._sleep_arm)
        self.ui.btn_calibrate.clicked.connect(partial(self.sm.set_next_state, "calibrate"))
        self.ui.btn_board_plane.clicked.connect(self._measure_board_plane)
        self.ui.btn_flatten_table.clicked.connect(self._flatten_table)
        self.ui.height_min.valueChanged.connect(self._height_band_changed)
        self.ui.height_max.valueChanged.connect(self._height_band_changed)
        self.ui.btn_depth_bias.clicked.connect(self._correct_depth_bias)
        self.ui.btn_clear_depth_bias.clicked.connect(self._clear_depth_bias)
        self.ui.btn_fixture.clicked.connect(self._measure_fixture)
        self.ui.btn_detector_measurement.clicked.connect(self._measure_detector)
        self.ui.chk_pick_place.stateChanged.connect(self._pick_place_chk)
        self.ui.btn_add_wp.clicked.connect(partial(self.sm.set_next_state, "add_waypoint"))
        self.ui.btn_clear_wps.clicked.connect(partial(self.sm.set_next_state, "clear_waypoints"))
        self.ui.btn_playback_wps.clicked.connect(partial(self.sm.set_next_state, "playback_waypoints"))

        # --- IK panel ---
        self._ik_result = None
        self._cartesian_jog_active = False
        # Column A is our closed form in sim, the vendor solver on the real arm;
        # column B is always our numerical solver.
        self.ui.ik_col_headers[0].setText('Geom' if self.arm_mode == "sim" else 'SDK')
        self.ui.ik_col_headers[1].setText('Num')
        self._clear_ik_results()
        self.ui.btn_ik_calc.clicked.connect(self._ik_calculate)
        self.ui.btn_ik_clear.clicked.connect(self._ik_clear)
        self.ui.btn_ik_goto.clicked.connect(self._ik_goto)
        for le in self.ui.ik_inputs.values():
            le.textChanged.connect(self._ik_inputs_changed)
            le.returnPressed.connect(self._ik_calculate)
        for btn, axis_idx, direction in self.ui.cart_jog_buttons:
            btn.pressed.connect(partial(self._start_cart_jog, axis_idx, direction))
            btn.released.connect(self._stop_cart_jog)

        self.ui.sldrMoveTime.valueChanged.connect(self._speed_slider_change)
        self._speed_slider_change()

        self.ui.chk_directcontrol.stateChanged.connect(self._direct_control_chk)
        self._set_cartesian_controls_enabled(True)

        for rb in (self.ui.radioVideo, self.ui.radioDepth, self.ui.radioTags, self.ui.radioWorkspace, self.ui.radioMask):
            rb.toggled.connect(self._render_current_frame)

        # --- Background threads ---
        self.sm_thread = StateMachineThread(self.sm)
        self.sm_thread.updateStatusMessage.connect(self._update_status_message)
        self.sm_thread.start()

        self.video_thread = VideoThread(self.camera)
        self.video_thread.updateFrame.connect(self._set_image)
        self.video_thread.start()

        self.arm_thread = SimArmThread(self.arm) if self.arm_mode == "sim" else ArmThread(self.arm)
        self.arm_thread.updateJointReadout.connect(self._update_joint_readout)
        self.arm_thread.updateEndEffectorReadout.connect(self._update_ee_readout)
        self.arm_thread.start()

        self._set_startup_status()
        # Enable buttons when arm is connected
        self.ui.btn_open_gripper.setEnabled(self.arm.connected)
        self.ui.btn_close_gripper.setEnabled(self.arm.connected)
        self.ui.btn_stop_gripper.setEnabled(self.arm.connected)
        if self.arm.connected:
            self.sm.set_next_state("initial_pose")

    # --- Startup ---

    def _measure_detector(self):
        from ui.detector_measurement import DetectorMeasurementDialog
        if not self.camera.camera_connected or self.camera.workspace_table_plane is None:
            self._set_status_message('Connect the camera, Calibrate and Flatten Table before measuring the detector.')
            return
        self.ui.radioWorkspace.setChecked(True)
        DetectorMeasurementDialog(self.camera, self).exec_()

    def _measure_fixture(self):
        from ui.fixture_measurement import FixtureMeasurementDialog
        if self.arm_mode != 'real' or not self.arm.connected or not self.arm.initialized:
            self._set_status_message('Fixture test requires an initialized real arm.')
            return
        if not self.camera.camera_connected or not self.camera.extrinsics_from_tags:
            self._set_status_message('Calibrate the camera from board tags before the fixture test.')
            return
        if self.sm.current_state != 'idle' or self.sm.next_state != 'idle':
            self._set_status_message('Finish the current arm operation before the fixture test.')
            return
        if self.ui.chk_directcontrol.isChecked() or self.ui.chk_manual_mode.isChecked() or self._cartesian_jog_active:
            self._set_status_message('Disable manual/direct control before the fixture test.')
            return
        FixtureMeasurementDialog(self.arm, self.camera, self).exec_()

    def _measure_board_plane(self):
        from ui.board_plane import BoardPlaneDialog
        if not self.camera.camera_connected:
            self._set_status_message("Camera offline - cannot measure board Z.")
            return
        BoardPlaneDialog(self.camera, self).exec_()

    def _flatten_table(self):
        from ui.table_plane import TablePlaneDialog
        if not self.camera.camera_connected or self.camera.workspace_homography is None:
            self._set_status_message('Calibrate the camera before flattening the table.')
            return
        if TablePlaneDialog(self.camera, self).exec_():
            self.ui.radioDepth.setChecked(True)
            self._set_status_message('Table flattened in Depth view. Recalibrate to clear the fit.')

    def _height_band_changed(self):
        low = self.ui.height_min.value()
        high = self.ui.height_max.value()
        if high <= low:
            self.ui.height_max.setValue(low + 1)
            return
        self.camera.block_height_band_mm = (low, high)

    def _correct_depth_bias(self):
        if not self.camera.camera_connected:
            self._set_status_message('Camera offline - cannot correct depth bias.')
            return
        try:
            info = self.camera.correct_depth_bias()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._set_status_message(str(exc))
            return
        self._set_status_message(
            f"Depth bias ON. Table samples: {info['inliers']}/{info['samples']}; "
            f"fit-frame depth RMS {info['before_rmse_mm']:.2f} -> {info['after_rmse_mm']:.2f} mm.")

    def _clear_depth_bias(self):
        self.camera.clear_depth_bias()
        self._set_status_message('Depth bias OFF. Restored the uncorrected table fit.')

    def _set_startup_status(self):
        """Show the initial camera and arm connection status."""
        parts = [
            "Camera online" if self.camera.camera_connected else "Camera offline",
            "Arm online"    if self.arm.connected           else "Arm offline",
        ]
        self._set_status_message(" | ".join(parts))
        if not self.camera.camera_connected:
            self.ui.videoDisplay.setText("No Video Input\n\nConnect the L515 camera\nvia USB 3.2")

    # --- Speed slider ---

    def _speed_slider_change(self):
        """Update the arm speed when the speed slider changes."""
        pct = self.ui.sldrMoveTime.value()
        self.ui.rdoutMoveTime.setText(f"{pct:d}%")
        self.arm.set_speed_pct(pct)

    # --- Video display ---

    def _selected_view_key(self):
        """Return the key for the currently selected camera view."""
        if self.ui.radioVideo.isChecked():  return "video"
        if self.ui.radioDepth.isChecked():  return "depth"
        if self.ui.radioTags.isChecked():   return "tags"
        if self.ui.radioMask.isChecked():   return "mask"
        return "workspace"

    def _render_current_frame(self):
        """Draw the latest frame for the selected camera view."""
        image = self._current_frames.get(self._selected_view_key())
        if image is not None:
            pm = QPixmap.fromImage(image)
            self.ui.videoDisplay.setPixmap(
                pm.scaled(self.ui.videoDisplay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def _sync_video_panel_geometry(self):
        """Keep the video display at a 16:9 size inside its panel."""
        # Fit the largest exact 16:9 video rectangle inside the space the
        # layout leaves for it, without cropping or stretching the feed.
        avail_w = max(self.ui.videoDisplay.width(), self.ui.videoDisplay.minimumWidth())
        avail_h = max(self.ui.videoDisplay.height(), self.ui.videoDisplay.minimumHeight())
        target_video_w = min(avail_w, int(round(avail_h * 16 / 9)))
        target_video_h = max(self.ui.videoDisplay.minimumHeight(), int(round(target_video_w * 9 / 16)))
        frame_padding = 16  # video card layout margins: 8px on each side
        self.ui.videoDisplay.setFixedSize(target_video_w, target_video_h)
        self.ui.videoCard.setFixedSize(target_video_w + frame_padding, target_video_h + frame_padding)
        self._render_current_frame()

    def _display_to_image_point(self, pos):
        """Convert a mouse position on the QLabel to an image pixel."""
        image = self._current_frames.get(self._selected_view_key())
        if image is None:
            return None
        lw, lh = self.ui.videoDisplay.width(), self.ui.videoDisplay.height()
        iw, ih = image.width(), image.height()
        scale = min(lw / iw, lh / ih)
        ox = (lw - int(iw * scale)) // 2
        oy = (lh - int(ih * scale)) // 2
        ix = (pos.x() - ox) / scale
        iy = (pos.y() - oy) / scale
        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
            return None
        return int(ix), int(iy)

    def _view_pixel_to_raw_image_pixel(self, x, y):
        """Map the selected view pixel to the raw image pixel used for depth."""
        if (self._selected_view_key() not in ("workspace", "depth", "mask")
                or self.camera.workspace_homography_inv is None):
            return x, y

        xy = self.camera.workspace_pixel_to_image(x, y)
        if xy is None:
            return x, y

        h, w = self.camera.depth_frame_raw.shape
        return (
            int(np.clip(round(xy[0]), 0, w - 1)),
            int(np.clip(round(xy[1]), 0, h - 1)),
        )

    # --- Jog controls ---

    def _set_direct_controls_enabled(self, enabled):
        """Enable or disable joint jog controls and the arm jog mode."""
        self.ui.SliderFrame.setProperty("active", enabled)
        self.ui.SliderControls.setEnabled(enabled)
        self._refresh_styles(self.ui.SliderFrame)
        if not (self.arm.connected and self.arm.initialized):
            return
        if enabled:
            self.arm.enter_jog_mode()
        else:
            self.arm.stop_jog()
            self.arm.enable()

    def _set_cartesian_controls_enabled(self, enabled):
        """Enable or disable Cartesian jog controls."""
        self.ui.CartesianJogFrame.setEnabled(enabled)
        self._refresh_styles(self.ui.CartesianJogFrame)
        if not enabled and self._cartesian_jog_active:
            self.arm.stop_cartesian_jog()
            self._cartesian_jog_active = False

    def _start_jog(self, joint_idx, direction):
        """Start jogging one joint while a jog button is held."""
        if not (self.arm.connected and self.arm.initialized):
            return
        if not self.arm.start_jog(joint_idx, direction):
            self._on_jog_fault()

    def _stop_jog(self):
        """Stop the active joint jog motion."""
        if self.arm.connected and self.arm.initialized and not self.arm.stop_jog():
            self._on_jog_fault()

    def _on_jog_fault(self):
        """Leave direct control and show a recovery message after a jog fault."""
        self.ui.chk_directcontrol.setChecked(False)
        self._set_status_message("Arm fault - check the arm is clear, then click Go Home to recover.")

    # --- Qt events ---

    def eventFilter(self, watched, event):
        """Handle mouse movement and clicks on the video display."""
        if watched is self.ui.videoDisplay:
            if event.type() == QEvent.MouseMove:
                self._track_mouse(event)
            elif event.type() == QEvent.MouseButtonPress:
                self._calibrate_mouse_press(event)
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        """Stop arm motion and worker threads before closing the window."""
        if self.arm.connected and self.arm.initialized:
            if self.ui.chk_directcontrol.isChecked():
                self.arm.stop_jog()
            if self._cartesian_jog_active:
                self.arm.stop_cartesian_jog()
        self.video_thread.stop()
        self.arm_thread.stop()
        self.sm_thread.stop()
        self.video_thread.wait(500)
        self.arm_thread.wait(500)
        self.sm_thread.wait(500)
        self.camera.stop()
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Resize the video panel after the main window layout changes."""
        self.ui.videoDisplay.setMinimumSize(320, 180)
        self.ui.videoDisplay.setMaximumSize(16777215, 16777215)
        self.ui.videoCard.setMinimumSize(0, 0)
        self.ui.videoCard.setMaximumSize(16777215, 16777215)
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_video_panel_geometry)

    # --- Slots: readouts ---

    @pyqtSlot(str)
    def _update_status_message(self, msg):
        """Display a status message in the GUI."""
        self.ui.rdoutStatus.setText(msg)

    def _set_status_message(self, msg):
        """Store and display the current state-machine status message."""
        self.sm.status_message = msg
        self._update_status_message(msg)

    def _refresh_styles(self, widget):
        """Force Qt to reapply stylesheet properties for a widget tree."""
        for w in [widget] + widget.findChildren(QWidget):
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()

    def _set_ik_result_label(self, label, text, state):
        """Set one IK result label and its visual state."""
        label.setText(text)
        label.setProperty("ikState", state)
        self._refresh_styles(label)

    def _clear_ik_results(self, invalid=False):
        """Clear both IK result columns and optionally mark them as invalid."""
        state = "invalid" if invalid else "neutral"
        for lbl in self.ui.ik_result_labels + self.ui.ik_result_labels_num:
            self._set_ik_result_label(lbl, '-', state)
        self._ik_result = None
        self.arm.set_ghost(None)   # no result -> no preview arm in the viewer
        self._update_ik_goto_enabled()

    def _show_ik_column(self, labels, result):
        """
        Fill one solver column and report which joints broke their limits.
        Returns the list of offending joint names ([] when the column is fine,
        None when the solver had no answer at all).
        """
        if result is None:
            for lbl in labels:
                self._set_ik_result_label(lbl, '-', "neutral")
            return None
        # result and joint_limits are both in radians; only the label is converted.
        violations = [
            JOINT_NAMES[idx]
            for idx, (angle, (lo, hi)) in enumerate(zip(result, self.arm.joint_limits))
            if angle < lo or angle > hi
        ]
        for idx, (angle, lbl) in enumerate(zip(result, labels)):
            state = "invalid" if JOINT_NAMES[idx] in violations else "valid"
            self._set_ik_result_label(lbl, f'{np.degrees(angle):+.2f}°', state)
        return violations

    def _update_ik_goto_enabled(self):
        """Enable Go To Pose only when the current IK result is usable."""
        enabled = (
            self._ik_result is not None
            and self.arm.connected
            and self.arm.initialized
            and not self.ui.chk_directcontrol.isChecked()
        )
        self.ui.btn_ik_goto.setEnabled(enabled)

    def _ik_inputs_changed(self, _text):
        """Clear stale IK results after the user edits the pose inputs."""
        had_result = self._ik_result is not None
        self._clear_ik_results()
        if had_result:
            self._set_status_message("IK: pose updated - recalculate to enable Go To Pose.")

    @pyqtSlot(list)
    def _update_joint_readout(self, joints):
        """Update joint angle labels and limit progress bars."""
        # joints and joint_limits are in radians; only the labels are converted.
        for idx, (rdout_card, rdout_jog, prog, joint) in enumerate(
            zip(self.ui.joint_readouts, self.ui.joint_jog_readouts, self.ui.joint_progs, joints)
        ):
            angle_deg = np.degrees(joint)
            rdout_card.setText(f"{angle_deg:+.1f}°")
            rdout_jog.setText(f"{angle_deg:+.1f}°")
            lo, hi = self.arm.joint_limits[idx]
            prog.setValue(max(0, min(100, int(100.0 * (joint - lo) / max(hi - lo, 1e-6)))))

    @pyqtSlot(list)
    def _update_ee_readout(self, pose):
        """Update the end-effector pose readout."""
        # pose is [x, y, z (mm), roll, pitch, yaw (rad)]; only the labels are converted.
        self.ui.rdoutX.setText(f"{pose[0]:+.2f} mm")
        self.ui.rdoutY.setText(f"{pose[1]:+.2f} mm")
        self.ui.rdoutZ.setText(f"{pose[2]:+.2f} mm")
        self.ui.rdoutPhi.setText(f"{np.degrees(pose[3]):+.2f}°")
        self.ui.rdoutTheta.setText(f"{np.degrees(pose[4]):+.2f}°")
        self.ui.rdoutPsi.setText(f"{np.degrees(pose[5]):+.2f}°")

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def _set_image(self, rgb_image, depth_image, tag_image, grid_image, mask_image):
        """Store new camera frames and render the selected one."""
        self.ui.depth_bias_status.setText(
            'Depth bias: ' + ('ON' if self.camera.active_depth_bias() is not None else 'OFF'))
        self._current_frames["video"]     = rgb_image
        self._current_frames["depth"]     = depth_image
        self._current_frames["tags"]      = tag_image
        self._current_frames["workspace"] = grid_image
        self._current_frames["mask"]      = mask_image
        self._render_current_frame()

    # --- Slots: arm actions ---

    def estop(self):
        """Stop the arm immediately and enter the estop state."""
        self.ui.chk_directcontrol.setChecked(False)  # triggers cleanup if direct control was on
        self.arm.stop()
        self.sm.set_next_state("estop")

    def _open_gripper(self):
        self.arm.open_gripper(sync=False)
        self.sm.gripper_state = "open"

    def _close_gripper(self):
        self.arm.close_gripper(sync=False)
        self.sm.gripper_state = "closed"

    def _set_manual_mode_panel_lock(self, locked):
        """Disable other panels while manual teach mode is active."""
        widgets = [
            self.ui.chk_directcontrol,
            self.ui.IKFrame,
            self.ui.SpeedFrame,
            self.ui.btn_initial_pose,
            self.ui.btn_sleep_arm,
            self.ui.btn_stop_gripper,
            # remove add wps, clear wps, open gripper, close gripper
            self.ui.btn_playback_wps,
            self.ui.btn_calibrate,
            self.ui.btn_board_plane,
            self.ui.btn_fixture,
            self.ui.chk_pick_place,
        ]
        if locked:
            self._pre_manual_widget_states = {w: w.isEnabled() for w in widgets}
            for widget in widgets:
                widget.setEnabled(False)
            self._set_cartesian_controls_enabled(False)
            self._refresh_styles(self.ui.IKFrame)
            return

        for widget, was_enabled in self._pre_manual_widget_states.items():
            widget.setEnabled(was_enabled)
        self._pre_manual_widget_states.clear()
        self._set_cartesian_controls_enabled(True)
        self._refresh_styles(self.ui.IKFrame)
        self._update_ik_goto_enabled()

    def _manual_mode_chk(self, state):
        """Turn manual teach mode on or off from the checkbox."""
        if state == Qt.Checked:
            if not self.arm.connected or not self.arm.initialized:
                self.ui.chk_manual_mode.blockSignals(True)
                self.ui.chk_manual_mode.setChecked(False)
                self.ui.chk_manual_mode.blockSignals(False)
                self._set_status_message(
                    "Arm offline - manual mode unavailable." if not self.arm.connected
                    else "Initialize the arm before enabling manual mode."
                )
                return
            if self.ui.chk_pick_place.isChecked():
                self.ui.chk_pick_place.setChecked(False)
            if self.ui.chk_directcontrol.isChecked():
                self.ui.chk_directcontrol.setChecked(False)
            self.arm.set_teach_mode(True)
            self._set_manual_mode_panel_lock(True)
            self._set_status_message("Manual mode is on - turn it off to use other panels.")
            return
        self.arm.set_teach_mode(False)
        self._set_manual_mode_panel_lock(False)
        self._set_status_message("Manual mode OFF.")

    def _direct_control_chk(self, state):
        """Turn joint direct-control mode on or off from the checkbox."""
        if state == Qt.Checked:
            if not self.arm.connected:
                self.ui.chk_directcontrol.setChecked(False)
                self._set_status_message("Arm offline - direct control unavailable.")
                return
            if not self.arm.initialized:
                self.ui.chk_directcontrol.setChecked(False)
                self._set_status_message("Initialize the arm before enabling direct control.")
                return
            self.sm.set_next_state("direct_control")
            self._set_cartesian_controls_enabled(False)
            self._set_direct_controls_enabled(True)
            self._update_ik_goto_enabled()
            return
        self.sm.set_next_state("idle")
        self._set_direct_controls_enabled(False)
        self._set_cartesian_controls_enabled(True)
        self._update_ik_goto_enabled()

    def _pick_place_chk(self, state):
        """Turn pick-and-place mode on or off from the checkbox."""
        if state == Qt.Checked:
            if not self.arm.connected or not self.arm.initialized:
                self.ui.chk_pick_place.setChecked(False)
                self._set_status_message("Initialize the arm before enabling Pick & Place.")
                return
            self.ui.chk_directcontrol.setChecked(False)
            self._stop_cart_jog()
            self.arm.enable()  # Pick/place joint moves require position mode.
            self.sm.set_next_state("pick_place")
            self._set_status_message("Pick & Place ON - click an object in the video feed.")
        else:
            self.sm.set_next_state("idle")

    def _initial_pose(self):
        """Leave direct control and request the initial arm pose."""
        self.ui.chk_directcontrol.setChecked(False)
        self.sm.set_next_state("initial_pose")

    def _sleep_arm(self):
        """Leave direct control and request the arm sleep pose."""
        self.ui.chk_directcontrol.setChecked(False)
        self.sm.set_next_state("sleep_arm")

    # --- IK panel ---

    def _ik_calculate(self):
        """Run IK for the requested pose and display the joint result."""
        if not self.arm.connected:
            self._clear_ik_results(invalid=True)
            self._set_status_message("IK: arm offline - connect the arm to use the solver.")
            return

        try:
            pose_display = [float(self.ui.ik_inputs[ax].text())
                            for ax in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')]
        except ValueError:
            self._clear_ik_results(invalid=True)
            self._set_status_message("IK: invalid input - enter numeric values for X/Y/Z and Roll/Pitch/Yaw.")
            return

        pose = pose_display[:3] + list(np.radians(pose_display[3:]))

        # Column A uses the SDK solver on the real arm and our closed form in
        # sim; column B is always our numerical solver, seeded from the current
        # joint angles. Both columns are shown so they can be compared.
        result_a = self.arm.get_ik(pose) if self.arm_mode == "sim" else self.arm.get_ik_sdk(pose)
        result_b = self.arm.get_ik_numerical(pose)
        self._ik_result = None
        self.arm.set_ghost(None)   # hide the stale preview; re-shown below if a solution lands

        violations_a = self._show_ik_column(self.ui.ik_result_labels, result_a)
        violations_b = self._show_ik_column(self.ui.ik_result_labels_num, result_b)

        if result_a is None and result_b is None:
            self._set_status_message("IK: no solution found for the requested end-effector pose.")
            self._update_ik_goto_enabled()
            return

        # Go To Pose prefers column A (predictable closed form / vendor solver)
        # and falls back to the numerical solution when A is unusable.
        name_a = 'geometric' if self.arm_mode == "sim" else 'SDK'
        if violations_a == []:
            self._ik_result = list(result_a)
            note = f"using the {name_a} solution"
        elif violations_b == []:
            self._ik_result = list(result_b)
            reason = "found none" if violations_a is None else "is outside the joint limits"
            note = f"using the numerical solution ({name_a} {reason})"
        else:
            self._update_ik_goto_enabled()
            broken = sorted(set((violations_a or []) + (violations_b or [])))
            names = ", ".join(broken)
            verb = "is" if len(broken) == 1 else "are"
            self._set_status_message(f"IK: solution found, but {names} {verb} outside the joint limits.")
            return

        # Preview the solution Go To Pose would actually run.
        self.arm.set_ghost(self._ik_result)
        self._update_ik_goto_enabled()
        if self.arm.initialized:
            self._set_status_message(f"IK: {note} - Go To Pose is enabled.")
        else:
            self._set_status_message(f"IK: {note} - initialize the arm to enable Go To Pose.")

    def _ik_clear(self):
        """Reset the IK panel: blank the pose inputs and both result columns."""
        # setText would fire _ik_inputs_changed six times and overwrite the
        # status message below, so mute the inputs while resetting them.
        for le in self.ui.ik_inputs.values():
            le.blockSignals(True)
            le.setText('0.0')
            le.blockSignals(False)
        self._clear_ik_results()
        self._set_status_message("IK: panel cleared.")

    def _ik_goto(self):
        """Move the arm to the most recent valid IK solution."""
        if self.ui.chk_directcontrol.isChecked():
            self.ui.chk_directcontrol.setChecked(False)
        self.arm.enable()  # restore mode 0 in case cartesian jog left arm in mode 5
        self.arm.set_ghost(None)  # the real arm is about to go there, so drop the preview
        self.arm.set_joint_angles(self._ik_result)
        self._set_status_message("Moving to IK pose...")

    def _start_cart_jog(self, axis_idx, direction):
        """Start jogging the end-effector along one Cartesian axis."""
        if not self.arm.connected:
            self._set_status_message("Arm offline - cartesian jog unavailable.")
            return
        if not self.arm.initialized:
            self._set_status_message("Initialize the arm before using Cartesian jog.")
            return
        if self.ui.chk_directcontrol.isChecked():
            self._set_status_message("Disable Direct Control before using Cartesian jog.")
            return
        self.arm.enter_cartesian_jog_mode()
        if not self.arm.start_cartesian_jog(axis_idx, direction, self.ui.radioEEFrame.isChecked()):
            self._cartesian_jog_active = False
            self._set_status_message("Arm fault during Cartesian jog - check the arm is clear.")
            return
        self._cartesian_jog_active = True

    def _stop_cart_jog(self):
        """Stop the active Cartesian jog motion."""
        if self.arm.connected and self.arm.initialized and self._cartesian_jog_active:
            self.arm.stop_cartesian_jog()
            self._cartesian_jog_active = False

    # --- Slots: mouse inspection ---

    def _track_mouse(self, mouse_event):
        """Update mouse pixel, depth, and world-coordinate readouts."""
        point = self._display_to_image_point(mouse_event.pos())
        if point is None:
            self.ui.rdoutMousePixels.setText("(u, v)")
            self.ui.rdoutMouseDepth.setText("-")
            self.ui.rdoutMouseWorld.setText("(uncalibrated)")
            return

        x, y = point
        ix, iy = self._view_pixel_to_raw_image_pixel(x, y)

        depth_mm = int(self.camera.depth_frame_raw[iy, ix] * self.camera.depth_scale_mm)
        self.ui.rdoutMousePixels.setText(f"({x:d}, {y:d})")
        self.ui.rdoutMouseDepth.setText(f"{depth_mm:d}" if depth_mm > 0 else "-")
        world = self.camera.image_to_world(ix, iy)
        self.ui.rdoutMouseWorld.setText(
            f"({world[0]:+.1f}, {world[1]:+.1f}, {world[2]:+.1f})" if world is not None else "(unavailable)"
        )

    def _calibrate_mouse_press(self, mouse_event):
        """Store the clicked image pixel for calibration or pick actions."""
        if mouse_event.button() == Qt.NoButton:
            return
        point = self._display_to_image_point(mouse_event.pos())
        if point is None:
            return
        x, y = point
        x, y = self._view_pixel_to_raw_image_pixel(x, y)
        self.camera.last_click[:] = [x, y]
        self.camera.new_click = True


def main():
    parser = argparse.ArgumentParser(description="Arm Lab control station")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sim", action="store_const", const="sim", dest="arm_mode", help="use the MuJoCo simulation arm (over LCM)")
    group.add_argument("--real", action="store_const", const="real", dest="arm_mode", help="connect to the real arm hardware (default)")
    parser.set_defaults(arm_mode="real")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app_window = Gui(arm_mode=args.arm_mode)
    app_window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
