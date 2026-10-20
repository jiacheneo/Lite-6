"""
MuJoCo simulation arm.

Instead of talking to the arm hardware, SimArm talks to the mujoco_sim MuJoCo
simulation over LCM. It is the mirror image of the bridge:
    * GET STATE  : subscribes to  ufactory_lite6_state   (ufactory_lite6_state_t)
    * SEND CONTROL: publishes to   ufactory_lite6_control (ufactory_lite6_control_t)
The MuJoCo model is position-controlled (`<position>` actuators, gains in the XML), so
SimArm only ever sends targets -- joint angles, joint velocities, or an end-effector
twist -- and reads sim state back. The bridge shapes those into a joint trajectory:
    * mode 0 : servo to qj_pos, speed/accel limited
    * mode 1 : integrate qj_vel (joint jog)
    * mode 2 : velocity IK on v_ee (Cartesian jog), the sim's stand-in for the real arm's firmware mode 5
SimArm exposes the same public API as Lite6Arm so control_station and state_machine can use either interchangeably (`--sim` vs `--real`).

Note:
    Gripper methods are stubs for now.
"""
import os
import sys
import threading
import time

import numpy as np

# Temp fix: add mujoco_sim to path
_MUJOCO_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_sim")
if _MUJOCO_SIM_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_SIM_DIR)

from PyQt5.QtCore import QThread, pyqtSignal

from kinematics import (DH_STD, JOINT_LIMITS, Q_DEFAULT, FK_dh, FK_pox, IK_geometric, IK_numerical, M, S_list, get_pose_from_T)

try:
    import lcm
    from lite6_sim.lcm_msgs import (ufactory_lite6_control_t, ufactory_lite6_display_t,
                                    ufactory_lite6_state_t)
    _LCM_AVAILABLE = True
except Exception as _exc:  # any import failure just disables the sim link
    _LCM_AVAILABLE = False
    _LCM_IMPORT_ERROR = _exc

STATE_CHANNEL = "ufactory_lite6_state"       # bridge -> us (sim state)
CONTROL_CHANNEL = "ufactory_lite6_control"   # us -> bridge (motion command)
DISPLAY_CHANNEL = "ufactory_lite6_display"   # us -> bridge (viewer overlays only)
LCM_URL = "udpm://239.255.76.67:7667?ttl=0"  # ttl=0: stays on this host, so stations do not cross-talk

# Control modes understood by the bridge (see ufactory_lite6_control_t.lcm).
MODE_POSITION = 0    # servo to qj_pos
MODE_JOINT_VEL = 1   # integrate qj_vel (joint jog)
MODE_CART_VEL = 2    # velocity IK on v_ee (Cartesian jog)

# Full-scale Cartesian jog rates, before the speed slider scales them. These match
# the caps the real arm's mode 5 accepts (lite6arm.start_cartesian_jog), so the two
# arms jog at the same speed for the same slider setting.
CART_JOG_LIN_MM_S = 150.0
CART_JOG_ROT_RAD_S = 1.0


class SimArm:
    """Simulation arm interface; drives the MuJoCo sim over LCM."""

    def __init__(self):
        self.connected = False
        self.initialized = False
        self.num_joints = 6
        self.joint_angles = [0.0] * self.num_joints
        # Lite 6 spec limits. The real arm reads these from firmware; the sim
        # has no firmware to query, so it uses the table in kinematics.py.
        self.joint_limits = [(lo, hi) for lo, hi in JOINT_LIMITS]
        self.dh_params = DH_STD  # standard Lite 6 DH table (sim has no firmware to query)
        self.speed_pct = 20.0
        self.mvacc_rad_s2 = 10.0
        # Direct-control (jog) state: per-joint velocity target (rad/s) streamed in velocity mode while a jog button is held.
        self._jog_active = False
        self._jog_vel = [0.0] * self.num_joints

        # LCM: subscribe to sim state, publish joint control.
        self._state_lock = threading.Lock()   # guards _latest_state
        self._pub_lock = threading.Lock()      # guards _latest_cmd + serializes publish
        self._latest_state = None              # last ufactory_lite6_state_t
        self._latest_cmd = None                # last ufactory_lite6_control_t (for re-streaming)

        # Viewer overlay state (published on DISPLAY_CHANNEL, no effect on the arm).
        self._ghost_q = None                   # IK preview joint angles (rad), None = hidden
        self._show_fk = True                   # draw the FK pose marker
        self._lc = None
        self._thread = None
        self._running = False
        if _LCM_AVAILABLE:
            try:
                self._lc = lcm.LCM(LCM_URL)
                self._lc.subscribe(STATE_CHANNEL, self._on_state)
                self._running = True
                self._thread = threading.Thread(target=self._lcm_loop, daemon=True)
                self._thread.start()
                self.connected = True
            except Exception as exc:
                print(f"WARNING: SimArm LCM unavailable ({exc})")
                self._lc = None
        else:
            print("WARNING: lite6_sim/lcm not importable. Sim arm disabled")

    def set_speed_pct(self, pct):
        self.speed_pct = float(np.clip(pct, 0, 100))

    def _speed_rad_s(self):
        # Joint speed (rad/s) from the speed slider, matching the real arm's mapping.
        return self.speed_pct / 100.0 * np.pi

    # --- LCM GET STATE (receiver) ---

    def _lcm_loop(self):
        while self._running:
            try:
                self._lc.handle_timeout(100)
            except Exception:
                pass

    def _on_state(self, channel, data):
        try:
            msg = ufactory_lite6_state_t.decode(data)
        except Exception:
            return
        with self._state_lock:
            self._latest_state = msg

    # --- LCM SEND CONTROL (publisher). Each command method calls _send_control. ---

    def _send_control(self, qj_pos=None, qj_vel=None, v_ee=None, ee_frame=False,
                      mode=MODE_POSITION, speed=0.0, mvacc=0.0):
        # Build and publish one control command; also store it so SimArmThread can re-stream it.
        # mode selects which of qj_pos / qj_vel / v_ee the bridge follows.
        # speed/mvacc: trajectory-shaping limits consumed by the bridge (rad/s, rad/s^2).
        # qj_tau/kp/kd stay zero: the sim is position-controlled, the servo gains live in the XML.
        if self._lc is None:
            return
        z = [0.0] * self.num_joints
        msg = ufactory_lite6_control_t()
        msg.timestamp = time.time_ns()
        msg.mode = int(mode)
        msg.speed = float(speed)
        msg.mvacc = float(mvacc)
        msg.qj_tau = list(z)
        msg.qj_pos = list(qj_pos) if qj_pos is not None else list(z)
        msg.qj_vel = list(qj_vel) if qj_vel is not None else list(z)
        msg.kp = list(z)
        msg.kd = list(z)
        msg.v_ee = list(v_ee) if v_ee is not None else [0.0] * 6
        msg.ee_frame = bool(ee_frame)
        with self._pub_lock:
            self._latest_cmd = msg
            try:
                self._lc.publish(CONTROL_CHANNEL, msg.encode())
            except Exception:
                pass

    def _republish_latest_cmd(self):
        # Re-stream (hold) the last command so the bridge keeps servoing the sim arm.
        # For a Cartesian jog this also keeps it running: the bridge re-solves the
        # velocity IK every sim step, so the same twist tracks the moving Jacobian.
        if self._lc is None:
            return
        with self._pub_lock:
            msg = self._latest_cmd
            if msg is None:
                return
            msg.timestamp = time.time_ns()
            try:
                self._lc.publish(CONTROL_CHANNEL, msg.encode())
            except Exception:
                pass

    # --- Viewer overlays (MuJoCo only - these never move the arm) ---

    def set_ghost(self, joint_angles_rad):
        # Show a translucent preview arm at these joint angles (rad); None hides it.
        # Published immediately so the viewer reacts to a button press without
        # waiting for the next SimArmThread tick.
        self._ghost_q = list(joint_angles_rad) if joint_angles_rad is not None else None
        self.publish_display()

    def set_fk_display(self, on):
        # Show or hide the FK pose marker (translucent box + axis triad).
        self._show_fk = bool(on)
        self.publish_display()

    def publish_display(self, ee_pose=None):
        # Publish the viewer overlay state. ee_pose is [x, y, z (mm), roll, pitch,
        # yaw (rad)] from the student FK; the wire format is SI, so mm -> m here.
        if self._lc is None:
            return
        if ee_pose is None:
            ee_pose = self.get_ee_pose()
        msg = ufactory_lite6_display_t()
        msg.timestamp = time.time_ns()
        msg.show_ghost = self._ghost_q is not None
        msg.ghost_qj_pos = list(self._ghost_q) if self._ghost_q is not None else [0.0] * self.num_joints
        msg.show_fk = self._show_fk
        msg.fk_pos = [v / 1000.0 for v in ee_pose[:3]]
        msg.fk_rpy = list(ee_pose[3:6])
        with self._pub_lock:
            try:
                self._lc.publish(DISPLAY_CHANNEL, msg.encode())
            except Exception:
                pass

    # --- Arm lifecycle ---

    def initialize(self):
        # Command the sim arm to the initial pose (speed/accel-limited servo move).
        if not self.connected:
            return False
        initial_pose = Q_DEFAULT  # matches UFactory Studio default; also the IK_numerical seed
        self._jog_active = False
        self._jog_vel = [0.0] * self.num_joints
        self._send_control(qj_pos=initial_pose, mode=MODE_POSITION,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        self.initialized = True
        return True

    def sleep(self):
        # NOTE: Unimplemented drive home + release
        self.initialized = False

    # --- Motion ---

    def set_joint_angles(self, joint_angles_rad, wait=False):
        # Send all 6 joint angles (rad) to the sim as a speed/accel-limited servo move.
        if not (self.connected and self.initialized):
            return False
        self._jog_active = False
        self._send_control(qj_pos=joint_angles_rad, mode=MODE_POSITION,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        if wait:
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                angles = self.get_joint_angles()
                if angles is not None and np.max(np.abs(
                        np.asarray(angles) - joint_angles_rad)) < np.radians(0.1):
                    return True
                time.sleep(0.02)
            return False
        return True

    # --- joint velocity (jog) mode: stream a velocity command the bridge integrates ---
    def enter_jog_mode(self):
        # Switch to joint-velocity mode; start held (zero velocity holds the current pose).
        self._jog_active = True
        self._jog_vel = [0.0] * self.num_joints
        self._send_control(qj_vel=self._jog_vel, mode=MODE_JOINT_VEL,
                           mvacc=self.mvacc_rad_s2)

    def start_jog(self, joint_idx, direction):
        # Command one joint's velocity; the bridge integrates it while it keeps arriving.
        if not (self.connected and self.initialized):
            return False
        self._jog_active = True
        self._jog_vel = [0.0] * self.num_joints
        self._jog_vel[joint_idx] = direction * self._speed_rad_s()
        self._send_control(qj_vel=self._jog_vel, mode=MODE_JOINT_VEL,
                           mvacc=self.mvacc_rad_s2)
        return True

    def stop_jog(self):
        # Zero the jog velocity; the bridge ramps to a stop and holds. Stay in velocity mode.
        if not (self.connected and self.initialized):
            return True
        self._jog_vel = [0.0] * self.num_joints
        self._send_control(qj_vel=self._jog_vel, mode=MODE_JOINT_VEL,
                           mvacc=self.mvacc_rad_s2)
        return True

    # --- Cartesian velocity (jog) mode: stream an EE twist the bridge runs velocity IK on ---
    def enter_cartesian_jog_mode(self):
        # Switch to Cartesian velocity mode; start held (a zero twist holds the current pose).
        self._jog_active = False
        self._jog_vel = [0.0] * self.num_joints
        self._send_control(v_ee=[0.0] * 6, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        return True

    def start_cartesian_jog(self, axis_idx, direction, in_ee_frame=False):
        # Jog one Cartesian axis: 0-2 are linear (mm/s), 3-5 angular (rad/s), same
        # scaling as the real arm so the speed slider means the same thing on both.
        vel = [0.0] * 6
        scale = CART_JOG_LIN_MM_S if axis_idx < 3 else CART_JOG_ROT_RAD_S
        vel[axis_idx] = direction * self.speed_pct / 100.0 * scale
        return self.set_cartesian_velocity(vel, in_ee_frame)

    def set_cartesian_velocity(self, vel, in_ee_frame=False):
        # Send a full 6-vector Cartesian velocity: vel = [vx, vy, vz, wrx, wry, wrz]
        # (mm/s, rad/s), matching the real arm's units. The wire format is SI, so mm -> m here.
        if not (self.connected and self.initialized):
            return False
        v_ee = [v / 1000.0 for v in vel[:3]] + list(vel[3:6])
        self._send_control(v_ee=v_ee, ee_frame=in_ee_frame, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)
        return True

    def stop_cartesian_jog(self):
        # Zero the twist; the bridge ramps to a stop and holds. Stay in Cartesian mode.
        if not (self.connected and self.initialized):
            return
        self._send_control(v_ee=[0.0] * 6, mode=MODE_CART_VEL,
                           speed=self._speed_rad_s(), mvacc=self.mvacc_rad_s2)

    def has_fault(self):
        return False  # sim has no hardware fault state (yet)

    def pause(self):
        pass  # NOTE: Unimplemented

    def resume(self):
        pass  # NOTE: Unimplemented

    def stop(self):
        pass  # NOTE: Unimplemented

    def enable(self):
        # Leave jog / Cartesian mode and hold the current pose (position mode, no move).
        self._jog_active = False
        self._jog_vel = [0.0] * self.num_joints
        if not (self.connected and self.initialized):
            return
        angles = self.get_joint_angles()
        if angles is None:
            return
        # speed=0 -> bridge tracks the target directly (re-syncs its desired pose to here).
        self._send_control(qj_pos=angles, mode=MODE_POSITION, speed=0.0)

    def set_teach_mode(self, on):
        pass  # NOTE: Unimplemented

    # --- Gripper (stubs - sim gripper not modeled yet) ---

    def open_gripper(self, wait=False, sync=True):
        pass  # NOTE: Unimplemented

    def close_gripper(self, wait=False, sync=True):
        pass  # NOTE: Unimplemented

    def stop_gripper(self, sync=True):
        pass  # NOTE: Unimplemented

    # --- Kinematics / state accessors ---

    def get_joint_angles(self):
        # Current joint angles (rad) from the latest received sim state, or None.
        with self._state_lock:
            state = self._latest_state
        if state is None:
            return None
        return list(state.qj_pos[:self.num_joints])

    def get_ee_pose_sdk(self):
        return [0.0] * 6  # NOT Used in sim

    def get_ik_sdk(self, pose):
        return None  # NOT Used in sim

    def get_ee_pose(self):
        # Student FK: base -> EE pose (mm, rad) from the latest joint angles.
        if self.dh_params is None:
            return [0.0] * 6
        # T = FK_dh(self.dh_params, self.joint_angles, self.num_joints)    # DH method
        T = FK_pox(self.joint_angles, M, S_list)                       # PoX method
        if T is None:
            return [0.0] * 6
        return get_pose_from_T(T)

    def get_ik(self, pose):
        # Student IK: return joint angles (rad) for the requested pose (mm, rad), or None if no solution.
        if self.dh_params is None:
            return None
        return IK_geometric(self.dh_params, pose)

    def get_ik_numerical(self, pose):
        # Student IK, numerical: seeded from the default (home) pose, not the
        # current one, so the same target always yields the same elbow-up answer.
        if self.dh_params is None:
            return None
        return IK_numerical(self.dh_params, pose, q0=Q_DEFAULT,
                            joint_limits=self.joint_limits)

    def stop_link(self):
        # Tear down the LCM receive thread.
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)


class SimArmThread(QThread):
    """Polls sim joint state at 20 Hz; EE pose at 10 Hz.

    Re-streams the latest control target each tick so the sim bridge keeps holding.
    """

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

            # Re-stream (hold) the last command.
            self.arm._republish_latest_cmd()

            angles = self.arm.get_joint_angles()
            if angles is not None:
                self.arm.joint_angles = angles
                self.updateJointReadout.emit(list(self.arm.joint_angles))
                if tick % 2 == 0:
                    ee_pose = self.arm.get_ee_pose()
                    self.updateEndEffectorReadout.emit(ee_pose)
                    # Same FK result drives the viewer's FK marker. Re-publishing
                    # every tick also lets a late-starting bridge catch up.
                    self.arm.publish_display(ee_pose)
            tick += 1
            time.sleep(0.05)


if __name__ == "__main__":
    arm = SimArm()
    print(f"SimArm connected={arm.connected}")
    arm.initialize()
    for _ in range(20):
        print("joint angles:", arm.get_joint_angles())
        time.sleep(0.1)
