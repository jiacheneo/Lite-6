"""
State machine for the arm lab runtime shell.
"""
import math
import time

from PyQt5.QtCore import QThread, pyqtSignal


class StateMachine:

    def __init__(self, arm, camera):
        self.arm = arm
        self.camera = camera
        self.status_message = "Idle - waiting for input."
        self.current_state = "idle"
        self.next_state = "idle"

        # Task 1.3: eight joint-space poses, stored in radians.
        # Start/end at Q_DEFAULT; sweep the base and roll the wrist.
        self.waypoints = [
            [math.radians(q) for q in pose]
            for pose in (
                (0, 9.9, 31.8, 0, 21.9, 0),
                (15, 9.9, 31.8, 0, 21.9, 0),
                (15, 9.9, 31.8, 0, 21.9, 20),
                (0, 9.9, 31.8, 0, 21.9, 0),
                (-15, 9.9, 31.8, 0, 21.9, 0),
                (-15, 9.9, 31.8, 0, 21.9, -20),
                (-15, 9.9, 31.8, 0, 21.9, 20),
                (0, 9.9, 31.8, 0, 21.9, 0),
            )
        ]
        self.recorded_waypoints = []
        self.gripper_state = "open"  # Assumed initial command state, not hardware feedback.
        self._held_block_height = None
        self.pick_place_block_height = 38.0  # Fallback without a matching detection.
        self.pick_place_clearance = 150.0  # Flange above clicked surface (mm).
        self._playback_index = 0
        self._playback_target = None
        self._playback_started = 0.0
        self.waypoint_tolerance = math.radians(1.0)
        self.waypoint_timeout = 120.0  # seconds per waypoint

        self._handlers = {
            "initial_pose":      self.initial_pose,
            "sleep_arm":         self.sleep_arm,
            "estop":             self.estop,
            "calibrate":         self.calibrate,
            "add_waypoint":      self.add_waypoint,
            "clear_waypoints":   self.clear_waypoints,
            "playback_waypoints":self.playback_waypoints,
            "direct_control":    self.direct_control,
            "pick_place":        self.pick_place,
        }

    def set_next_state(self, state):
        self.next_state = state

    def _go_idle(self, message):
        self.status_message = message
        self.current_state = "idle"
        self.next_state = "idle"

    def run(self):
        self._handlers.get(self.next_state, self.idle)()

    def idle(self):
        if self.current_state != "idle":
            self.status_message = "Idle - waiting for input."
        self.current_state = "idle"

    def direct_control(self):
        if not self.arm.connected or not self.arm.initialized:
            self._go_idle("Direct control unavailable until the arm is initialized.")
            return
        if self.current_state != "direct_control":
            self.status_message = "Direct Control - use jog controls to move the arm."
        self.current_state = "direct_control"

    def estop(self):
        if self.current_state != "estop":
            self.arm.stop()
            self.status_message = "STOP active - clear the arm and click Go Home to recover."
            self.current_state = "estop"

    def initial_pose(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot initialize.")
            return
        self.status_message = "Initializing arm..."
        if not self.arm.initialize():
            self._go_idle("Failed to initialize arm.")
        else:
            self._go_idle("Arm initialized - at initial pose.")

    def sleep_arm(self):
        if not self.arm.connected:
            self._go_idle("Arm offline - cannot send Sleep.")
            return
        self.status_message = "Sleep: returning to home position, motors will shut off..."
        self.arm.sleep()
        self._go_idle("Sleep: home position reached, motors off.")

    def calibrate(self):
        _, message = self.camera.estimate_extrinsics_from_tags()
        self._go_idle(message)

    def add_waypoint(self):
        # Record the last gripper state requested through the GUI.
        self.recorded_waypoints.append({
            "angles": list(self.arm.get_joint_angles()),  # radians
            "gripper": self.gripper_state,
        })
        self._go_idle(f"Recorded waypoint {len(self.recorded_waypoints)}.")

    def clear_waypoints(self):
        self.recorded_waypoints.clear()
        self._go_idle("Recorded waypoints cleared.")

    def playback_waypoints(self):
        self.current_state = "playback_waypoints"
        for waypoint in list(self.recorded_waypoints):
            if self.next_state != "playback_waypoints":
                return
            self.arm.set_joint_angles(waypoint["angles"])

            # Motion commands return immediately, so wait for arrival.
            while True:
                if self.next_state != "playback_waypoints":
                    return
                angles = self.arm.get_joint_angles()
                if all(
                    abs(actual - target) <= self.waypoint_tolerance
                    for actual, target in zip(angles, waypoint["angles"])
                ):
                    break
                time.sleep(0.05)

            self.gripper_state = waypoint["gripper"]
            if self.gripper_state == "closed":
                self.arm.close_gripper()
            else:
                self.arm.open_gripper()
            time.sleep(1.2)

        if self.next_state == "playback_waypoints":
            self._go_idle("Playback complete.")

    def pick_place(self):
        """Alternate a top-down pick and place, using the clicked surface depth."""
        if not self.arm.connected or not self.arm.initialized:
            self._go_idle("Initialize the arm before enabling Pick & Place.")
            return
        if self.current_state != "pick_place":
            self.current_state = "pick_place"
            self.camera.new_click = False  # Ignore clicks made before enabling.
            self.status_message = self._pick_place_prompt()
            return
        if not self.camera.new_click:
            return

        u, v = self.camera.last_click
        self.camera.new_click = False
        point = self.camera.image_to_world(u, v)
        if point is None or not all(math.isfinite(value) for value in point):
            self.status_message = "Pick & Place - invalid depth/calibration; click again."
            return
        # A manual Open Gripper command also ends a pending place operation.
        if self.gripper_state == "open":
            self._held_block_height = None
        x, y, surface_z = point
        placing = self._held_block_height is not None
        block_height = self._held_block_height if placing else self.pick_place_block_height
        if not placing:
            # Use detector size when clicking near a detected block center.
            for detection in self.camera.block_detections:
                center = detection.get("world_position_mm")
                size = detection.get("nominal_size_mm", self.pick_place_block_height)
                if center is not None and math.hypot(x - center[0], y - center[1]) <= size / 2:
                    block_height = size
                    break

        # A pick click measures the block TOP; a place click measures its support.
        center_z = surface_z + (block_height / 2 if placing else -block_height / 2)
        target = [x, y, center_z + 85.0, math.pi, 0.0, 0.0]
        current = list(self.arm.get_ee_pose())
        travel_z = max(current[2], surface_z + self.pick_place_clearance)
        above = [x, y, travel_z, math.pi, 0.0, 0.0]

        # Command each endpoint once; no intermediate descent/lift waypoints.
        # These are joint moves, so the tool path is not necessarily vertical.
        lifted = current.copy()
        lifted[2] = travel_z
        poses = [lifted, above, target]
        joints = [self.arm.get_ik(pose) for pose in poses]
        for i, (pose, q) in enumerate(zip(poses, joints)):
            if q is None:
                stage = ("initial lift", "approach", "descent")[i]
                self.status_message = (
                    f"Pick & Place - {stage} unreachable at XYZ "
                    f"({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:.0f}) mm; click again.")
                return  # Validate the entire approach before issuing any commands.

        if self.next_state != "pick_place":
            return
        action = "Placing" if placing else "Picking"
        self.status_message = f"{action} at ({x:.0f}, {y:.0f}) mm..."
        try:
            if not placing:
                self.arm.open_gripper(wait=True)
                self.gripper_state = "open"
            for q in joints:
                if not self._pick_place_move(q):
                    return
            if self.next_state != "pick_place":
                return
            if placing:
                self.arm.open_gripper(wait=True)
                self.gripper_state = "open"
                self._held_block_height = None
            else:
                self.arm.close_gripper(wait=True)
                self.gripper_state = "closed"
                self._held_block_height = block_height

            # Lift directly back to the approach pose in one move.
            if not self._pick_place_move(joints[1]):
                return
            self.status_message = self._pick_place_prompt()
        finally:
            self.camera.new_click = False  # Discard clicks made during motion.

    def _pick_place_prompt(self):
        if self._held_block_height is not None:
            return "Pick & Place - holding block; click a table/stack location to place."
        return "Pick & Place - click the center of a block to pick."

    def _pick_place_move(self, joints):
        if self.next_state != "pick_place":
            return False
        if not self.arm.set_joint_angles(joints, wait=True):
            if self.next_state == "pick_place":
                self._go_idle("Pick & Place - motion failed; check arm and re-enable mode.")
            return False
        return self.next_state == "pick_place"


class StateMachineThread(QThread):
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        QThread.__init__(self, parent=parent)
        self.sm = state_machine
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
