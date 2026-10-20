"""
Lite 6 arm wrapper around the xArm Python SDK.
"""
import time

import numpy as np

# Temp fix: add mujoco_sim to path
import os
import sys
_MUJOCO_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_sim")
if _MUJOCO_SIM_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_SIM_DIR)

from PyQt5.QtCore import QThread, pyqtSignal
from xarm.core.config.x_config import XCONF
from xarm.wrapper import XArmAPI
from kinematics import DH_STD, FK_dh, get_pose_from_T, IK_geometric, IK_numerical, Q_DEFAULT

try:
    import lcm
    from lite6_sim.lcm_msgs import ufactory_lite6_state_t
    _LCM_AVAILABLE = True
except Exception as _exc:
    _LCM_AVAILABLE = False  # any import failure should just disable replay
    _LCM_IMPORT_ERROR = _exc

XARM_IP = "192.168.1.152"  # TODO: replace this with your station's arm IP
JOINT_NAMES = ("Base", "Shoulder", "Elbow", "F.Roll", "W.Pitch", "W.Roll")

LCM_STATE_CHANNEL = "ufactory_lite6_state"  # must match mujoco_sim bridge.topic_state
LCM_URL = "udpm://239.255.76.67:7667?ttl=0"  # ttl=0: stays on this host, so stations do not cross-talk

GRIPPER_DWELL_S = 0.8  # Gripper Lite (LG-1000) has no feedback; dwell to let it finish actuating

class Lite6Arm:
    """Wrapper around XArmAPI for the UFactory Lite 6."""

    def __init__(self, ip=XARM_IP):
        self.connected = False
        self.initialized = False
        self.num_joints = 6
        self.joint_angles = [0.0] * self.num_joints
        self.joint_limits = [(-np.pi, np.pi) for _ in range(self.num_joints)] # fallback
        self.dh_params = None
        self.speed_pct = 20.0
        self.mvacc_rad_s2 = 10.0
        self.xarm = None

        try:
            self.xarm = XArmAPI(ip, is_radian=True) # all input/output angles in radian
            self.connected = True
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(0)
            self.xarm.set_state(state=0)

            self.joint_limits = XCONF.Robot.JOINT_LIMITS[self.num_joints][self.xarm.device_type]
            # Student FK/IK require six-row standard DH. The SDK's raw seven-row
            # table shifts alpha/a and cannot be passed to these solvers directly.
            self.dh_params = DH_STD.copy()
        except Exception as exc:
            print(f"WARNING: Arm unavailable ({exc})")

        # LCM publisher: broadcast joint angles
        self.lcm_pub = None
        self._lcm_state_msg = None
        if _LCM_AVAILABLE:
            try:
                self.lcm_pub = lcm.LCM(LCM_URL)
                self._lcm_state_msg = ufactory_lite6_state_t()
            except Exception as exc:
                print(f"WARNING: LCM publisher unavailable ({exc})")
        else:
            print("WARNING: lite6_sim/lcm not importable. MuJoCo replay disabled")

    def set_speed_pct(self, pct):
        self.speed_pct = float(np.clip(pct, 0, 100))

    # --- Arm lifecycle ---

    def initialize(self):
        # Enable motors, configure collision detection, move to initial pose.
        if not self.connected:
            return False
        self.initialized = False
        self.enable()
        self.xarm.set_self_collision_detection(True)
        self.xarm.set_collision_sensitivity(5)
        initial_pose = Q_DEFAULT  # matches UFactory Studio default; also the IK_numerical seed
        self.xarm.set_servo_angle(angle=initial_pose, speed=0.1 * np.pi, mvacc=self.mvacc_rad_s2, wait=True)
        self.initialized = True
        return True

    def sleep(self):
        # Restore position mode, return to mechanical zero, then cut motor power.
        self.enable()
        self.xarm.move_gohome(speed=0.1 * np.pi, mvacc=self.mvacc_rad_s2, wait=True)
        self.xarm.motion_enable(enable=False)
        self.initialized = False

    # --- Motion ---

    def set_joint_angles(self, joint_angles_rad, wait=False):
        # Send all 6 joint angles (rad); optionally wait for arrival.
        if not self.connected or not self.initialized:
            return False
        return self.xarm.set_servo_angle(
            angle=joint_angles_rad,
            speed=self.speed_pct / 100.0 * np.pi,
            mvacc=self.mvacc_rad_s2,
            wait=wait,
        ) == 0

    def enter_jog_mode(self):
        # Switch to velocity control once when direct control is enabled.
        self.xarm.set_mode(4)
        self.xarm.set_state(0)

    def start_jog(self, joint_idx, direction):
        # Velocity mode already active - just update velocity. Returns False on arm fault.
        velocities = [0.0] * self.num_joints
        velocities[joint_idx] = direction * self.speed_pct / 100.0 * np.pi
        return self.xarm.vc_set_joint_velocity(velocities, is_radian=True) == 0

    def stop_jog(self):
        # Zero velocity - stay in velocity mode until direct control is disabled. Returns False on fault.
        return self.xarm.vc_set_joint_velocity([0.0] * self.num_joints, is_radian=True) == 0

    def enter_cartesian_jog_mode(self):
        # Switch to Cartesian velocity control (mode 5). Returns False on fault.
        ok = self.xarm.set_mode(5) == 0
        ok = (self.xarm.set_state(0) == 0) and ok
        return ok

    def start_cartesian_jog(self, axis_idx, direction, in_ee_frame=False):
        # Cartesian velocity mode - linear axes in mm/s (max 150), angular in rad/s (max 1.0).
        vel = [0.0] * 6
        vel[axis_idx] = direction * self.speed_pct / 100.0 * (150.0 if axis_idx < 3 else 1.0)
        return self.xarm.vc_set_cartesian_velocity(vel, is_radian=True, is_tool_coord=in_ee_frame) == 0

    def set_cartesian_velocity(self, vel, in_ee_frame=False):
        # Send a full 6-vector Cartesian velocity: vel = [vx, vy, vz, wrx, wry, wrz] (mm/s, rad/s). Returns True on success (SDK code 0).
        if not (self.connected and self.initialized):
            return False
        return self.xarm.vc_set_cartesian_velocity(vel, is_radian=True, is_tool_coord=in_ee_frame) == 0

    def stop_cartesian_jog(self):
        self.xarm.vc_set_cartesian_velocity([0.0] * 6, is_radian=True)

    def has_fault(self):
        # True if the arm is reporting an error (collision / joint limit / e-stop / fault).
        if not self.connected:
            return False
        try:
            return bool(self.xarm.has_error)
        except Exception:
            return False

    def pause(self):
        # Suspend motion mid-move; call resume() to continue.
        if self.connected:
            self.xarm.set_state(state=3)

    def resume(self):
        # Resume after pause().
        if self.connected:
            self.xarm.set_state(state=0)

    def stop(self):
        # Hard stop - motors stay energized but frozen. Call enable() to restore motion.
        if self.connected:
            self.xarm.set_state(state=4)

    def enable(self):
        # Restore full normal operation (motion_enable + mode 0 + state 0).
        if not self.connected:
            return
        self.xarm.motion_enable(enable=True)
        self.xarm.set_mode(0)
        self.xarm.set_state(state=0)

    def set_teach_mode(self, on):
        # on=True: drag mode. on=False: back to normal.
        if not self.connected:
            return
        if on:
            self.xarm.set_mode(2)
            self.xarm.set_state(state=0)
        else:
            self.enable()

    # --- Gripper ---

    def open_gripper(self, wait=False, sync=True):
        # Open the Lite 6 Gripper Lite (LG-1000).
        # Binary open/close only, no position feedback. sync=False actuates immediately in velocity mode
        if not self.connected:
            return
        code = self.xarm.open_lite6_gripper(sync=sync)
        if code != 0:
            print(f"WARNING: open_lite6_gripper failed (code={code})")
        elif wait:
            time.sleep(GRIPPER_DWELL_S)

    def close_gripper(self, wait=False, sync=True):
        # Close (grip) the Lite 6 Gripper Lite.
        if not self.connected:
            return
        code = self.xarm.close_lite6_gripper(sync=sync)
        if code != 0:
            print(f"WARNING: close_lite6_gripper failed (code={code})")
        elif wait:
            time.sleep(GRIPPER_DWELL_S)

    def stop_gripper(self, sync=True):
        # Turn off the Lite 6 Gripper Lite drive (pump off).
        if not self.connected:
            return
        code = self.xarm.stop_lite6_gripper(sync=sync)
        if code != 0:
            print(f"WARNING: stop_lite6_gripper failed (code={code})")

    # --- Kinematics ---

    def get_joint_angles(self):
        # Returns current joint angles (rad) or None on SDK error.
        code, angles = self.xarm.get_servo_angle()
        if code == 0:
            return angles[:self.num_joints]
        return None

    # --- LCM state broadcast ---

    def publish_lcm_state(self, joint_angles_rad):
        # Publish joint angles (rad) on the state channel
        if self.lcm_pub is None:
            return
        self._lcm_state_msg.timestamp = time.time_ns()
        self._lcm_state_msg.qj_pos = list(joint_angles_rad[:self.num_joints])
        self.lcm_pub.publish(LCM_STATE_CHANNEL, self._lcm_state_msg.encode())

    def set_ghost(self, joint_angles_rad):
        # No-op: the ghost preview is a MuJoCo viewer overlay, so it only exists
        # in sim. Defined here so the GUI can call it without checking arm mode.
        pass

    def set_fk_display(self, on):
        # No-op on hardware; see set_ghost.
        pass

    def get_ee_pose_sdk(self):
        code, pose = self.xarm.get_forward_kinematics(self.joint_angles)
        if code == 0:
            return pose
        print(f"WARNING: SDK FK failed (code={code})")
        return [0.0] * 6

    def get_ik_sdk(self, pose):
        if not self.connected:
            return None
        code, angles = self.xarm.get_inverse_kinematics(pose)
        if code == 0:
            return angles
        print(f"WARNING: SDK IK failed (code={code})")
        return None

    def get_ee_pose(self):
        """
        Student lab: implement FK_dh and get_pose_from_T in kinematics.py.
        """
        if self.dh_params is None:
            return [0.0] * 6
        T = FK_dh(self.dh_params, self.joint_angles, self.num_joints)
        if T is None:
            return [0.0] * 6
        return get_pose_from_T(T)

    def get_ik(self, pose):
        """
        Student lab: implement IK in kinematics.py.
        """
        if self.dh_params is None:
            return None
        return IK_geometric(self.dh_params, pose)

    def get_ik_numerical(self, pose):
        """
        Student lab: numerical IK, seeded from the default (home) pose rather
        than the arm's current angles so the same target always yields the same
        elbow-up answer, and bounded by the joint limits read from firmware.
        """
        if self.dh_params is None:
            return None
        return IK_numerical(self.dh_params, pose, q0=Q_DEFAULT,
                            joint_limits=self.joint_limits)


class ArmThread(QThread):
    """Polls joint angles at 20 Hz; EE pose at 10 Hz (every other tick)."""

    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, arm, parent=None):
        QThread.__init__(self, parent=parent)
        self.arm = arm
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        tick = 0
        while self._running:
            if not self.arm.connected:
                time.sleep(0.25)
                continue

            angles = self.arm.get_joint_angles()
            if angles is not None:
                self.arm.joint_angles = angles
                self.arm.publish_lcm_state(angles)
                self.updateJointReadout.emit(self.arm.joint_angles)
                if tick % 2 == 0:
                    self.updateEndEffectorReadout.emit(self.arm.get_ee_pose_sdk())
            tick += 1
            time.sleep(0.05)


if __name__ == '__main__':
    print(f"Connecting to Lite 6 at {XARM_IP} ...")
    arm = Lite6Arm()
    if not arm.connected:
        print("Arm unavailable.")
    else:
        print("Initializing...")
        arm.initialize()
        code, angles = arm.xarm.get_servo_angle()
        print(f"Joint angles at home (rad): {[f'{a:.3f}' for a in angles[:6]]}")
        print(f"EE pose from SDK FK: {[f'{v:.4f}' for v in arm.get_ee_pose_sdk()]}")
        time.sleep(2)
        print("Sleeping arm...")
        arm.sleep()
        print("Done.")
