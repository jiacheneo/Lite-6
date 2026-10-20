"""
Kinematics for the UFactory Lite 6.

TODO: Implement all functions marked with TODO below.
      For FK, implement either the DH method or the PoX method (your choice).
      For IK, implement IK_geometric.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ======================================================================
# FK - DH Method
# ======================================================================

# Standard DH parameters for the UFactory Lite 6: https://docs.supportarticle.ufactory.cc/support_articles/developer/kinematic-and-dynamic-parameters/lite6.html
# One row per joint: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
DH_STD = np.array([
    [0.0,        243.3, -np.pi / 2,   0.0],  # Joint 1
    [-np.pi / 2,   0.0,  np.pi,     200.0],  # Joint 2
    [-np.pi / 2,   0.0,  np.pi / 2,  87.0],  # Joint 3
    [0.0,        227.6,  np.pi / 2,   0.0],  # Joint 4
    [0.0,          0.0, -np.pi / 2,   0.0],  # Joint 5
    [0.0,         61.5,  0.0,         0.0],  # Joint 6
], dtype=float)

# Lite 6 joint travel limits, matching the MuJoCo model
# (robot_assets/UfactoryLite6/xml/lite6.xml). Joint 3 is asymmetric.
JOINT_LIMITS_DEG = np.array([
    [-360.0, 360.0],   # Joint 1
    [-150.0, 150.0],   # Joint 2
    [  -3.5, 300.0],   # Joint 3
    [-360.0, 360.0],   # Joint 4
    [-124.0, 124.0],   # Joint 5
    [-360.0, 360.0],   # Joint 6
], dtype=float)
JOINT_LIMITS = np.radians(JOINT_LIMITS_DEG)   # (6, 2) in radians

# The arm's default (home) configuration, matching the UFactory Studio default.
# This is both the pose Initial Pose drives to and the seed IK_numerical starts from.
Q_DEFAULT_DEG = np.array([0.0, 9.9, 31.8, 0.0, 21.9, 0.0], dtype=float)
Q_DEFAULT = np.radians(Q_DEFAULT_DEG)         # (6,) in radians


def get_transform_from_dh(theta_offset, d, alpha, a, joint_angle):
    """
    Build the 4x4 transform for one DH row.

    Standard-DH row format (from arm.dh_params):
        theta_offset (rad), d (mm), alpha (rad), a (mm)
    joint_angle: current joint angle in radians.
    """
    theta = theta_offset + joint_angle
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # Standard DH: Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha).
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0],
    ], dtype=float)


def FK_dh(dh_params, joint_angles_rad, num_joints):
    """
    Forward kinematics via DH convention.

    Calls get_transform_from_dh for each joint and chains the results.

    dh_params:        Standard-DH table, flat list or an (n, 4) array.
                      Raw SDK get_dh_params() uses a different layout; convert it first.
                      Each row: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
    joint_angles_rad: joint angles in radians, length num_joints.
    num_joints:       number of joints to chain (pass 6 for end-effector).

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    dh_params = np.asarray(dh_params, dtype=float).reshape(-1, 4)
    T = np.eye(4)

    for i in range(num_joints):
        theta_offset, d, alpha, a = dh_params[i]
        A = get_transform_from_dh(
            theta_offset, d, alpha, a, joint_angles_rad[i]
        )
        T = T @ A

    return T


# ======================================================================
# FK - PoX Method
# ======================================================================

# M and S_list are robot constants for the PoX method:
#   M:      4x4 end-effector transform at the zero (home) configuration.
#   S_list: 6x6 - one space-frame screw axis per row: [w1, w2, w3, v1, v2, v3]
# Units are mm, like FK_dh.

# Zero joint angles (not Q_DEFAULT): flange at (87, 0, 154.2) mm,
# with its x axis along base +x, y along -y, and z along -z.
M = np.array([
    [1,  0,  0,  87.0],
    [0, -1,  0,   0.0],
    [0,  0, -1, 154.2],
    [0, 0, 0, 1],
], dtype=float)

S_list = np.array([
    # wx  wy  wz     vx     vy    vz
    [ 0,  0,  1,    0.0,   0.0,  0.0],  # Joint 1
    [ 0,  1,  0, -243.3,   0.0,  0.0],  # Joint 2
    [ 0, -1,  0,  443.3,   0.0,  0.0],  # Joint 3
    [ 0,  0, -1,    0.0,  87.0,  0.0],  # Joint 4
    [ 0,  1,  0, -215.7,   0.0, 87.0],  # Joint 5
    [ 0,  0, -1,    0.0,  87.0,  0.0],  # Joint 6
], dtype=float)


def to_s_matrix(w, v):
    """
    Build the 4x4 twist matrix [S] with a skew-symmetric angular block.
    w: (3,) angular velocity component
    v: (3,) linear velocity component
    """
    wx, wy, wz = w
    vx, vy, vz = v
    return np.array([
        [0.0, -wz,  wy, vx],
        [ wz, 0.0, -wx, vy],
        [-wy,  wx, 0.0, vz],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=float)


def FK_pox(joint_angles_rad, m_mat, s_lst):
    """
    Forward kinematics via product of exponentials.

    Calls to_s_matrix for each joint and chains e^([S]*theta).

    joint_angles_rad: raw joint angles in radians; M and s_lst encode the zero configuration.
    m_mat:            4x4 home configuration matrix M.
    s_lst:            (6, 6) screw axes, one per row: [w1, w2, w3, v1, v2, v3].

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    s_lst = np.asarray(s_lst, dtype=float)
    T = np.eye(4)
    for i, angle in enumerate(joint_angles_rad):
        S = to_s_matrix(s_lst[i, :3], s_lst[i, 3:])
        T = T @ expm(S * angle)
    return T @ np.asarray(m_mat, dtype=float)


# ======================================================================
# FK - shared output helper
# ======================================================================

def get_pose_from_T(T):
    """
    Extract [x, y, z, phi, theta, psi] from a 4x4 homogeneous transform.
    x, y, z  in mm.  phi, theta, psi  in radians.

    Orientation is roll-pitch-yaw about fixed X, Y, Z axes (matches the
    xArm SDK convention and the GUI's Roll/Pitch/Yaw readout).

    Used by both FK_dh and FK_pox to produce the pose vector for the GUI.
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    # Lowercase xyz selects rotations about fixed axes, in radians.
    roll, pitch, yaw = Rotation.from_matrix(rotation_matrix).as_euler(
        'xyz', degrees=False
    )
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(roll),
        float(pitch),
        float(yaw),
    ]


# ======================================================================
# IK - Geometric Method
# ======================================================================

def IK_geometric(dh_params, pose):
    """
    Inverse kinematics for the Lite 6.

    pose: [x, y, z, phi, theta, psi]  (mm, radians).

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.

    Prefer the shoulder, elbow and wrist branches containing Q_DEFAULT;
    try the other branches only when joint limits rule those out.
    """
    dh = np.asarray(dh_params, dtype=float).reshape(-1, 4)[:6]
    pose = np.asarray(pose, dtype=float)
    offsets = dh[:, 0]
    R = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    wrist = pose[:3] - dh[5, 1] * R[:, 2]

    upper_arm = dh[1, 3]
    forearm = np.hypot(dh[2, 3], dh[3, 1])
    elbow_offset = np.arctan2(dh[3, 1], dh[2, 3])
    radius = np.hypot(wrist[0], wrist[1])
    height = dh[0, 1] - wrist[2]
    cosine = (radius**2 + height**2 - upper_arm**2 - forearm**2) / (
        2 * upper_arm * forearm
    )
    tol = 1e-9
    if abs(cosine) > 1 + tol:
        return None
    elbow_angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    # At the base axis, joint 1 is free. Start at home; the tool-axis
    # azimuth is a fallback when the home choice exceeds wrist limits.
    if radius < tol:
        azimuths = [Q_DEFAULT[0] + offsets[0], np.arctan2(R[1, 2], R[0, 2])]
        azimuths += [azimuths[1] + np.pi]
    else:
        azimuth = np.arctan2(wrist[1], wrist[0])
        azimuths = [azimuth, azimuth + np.pi]

    lo, hi = JOINT_LIMITS.T
    for azimuth in azimuths:
        radial = wrist[0] * np.cos(azimuth) + wrist[1] * np.sin(azimuth)
        for delta in (elbow_angle, -elbow_angle):
            # In the shoulder plane, the effective forearm angle relative
            # to the upper arm is delta = elbow_offset - theta3.
            q = np.zeros(6)
            q[0] = azimuth - offsets[0]
            q[1] = (np.arctan2(height, radial)
                    - np.arctan2(forearm * np.sin(delta),
                                 upper_arm + forearm * np.cos(delta))
                    - offsets[1])
            q[2] = elbow_offset - delta - offsets[2]

            R03 = FK_dh(dh, q, 3)[:3, :3]
            R36 = R03.T @ R
            sine = np.hypot(R36[0, 2], R36[1, 2])
            theta5 = np.arctan2(sine, R36[2, 2])
            if sine < tol:
                # At theta5 = 0 only theta4 + theta6 is determined.
                theta4 = Q_DEFAULT[3] + offsets[3]
                if R36[2, 2] > 0:
                    theta6 = np.arctan2(R36[1, 0], R36[0, 0]) - theta4
                else:
                    theta6 = theta4 - np.arctan2(-R36[1, 0], -R36[0, 0])
                wrists = [(theta4, theta5, theta6)]
            else:
                # This DH wrist is Rz(theta4) Ry(-theta5) Rz(theta6).
                theta4 = np.arctan2(-R36[1, 2], -R36[0, 2])
                theta6 = np.arctan2(-R36[2, 1], R36[2, 0])
                wrists = [(theta4, theta5, theta6),
                          (theta4 + np.pi, -theta5, theta6 + np.pi)]

            for wrist_angles in wrists:
                q[3:] = np.asarray(wrist_angles) - offsets[3:]
                # Equivalent 2*pi turns matter for joint 3's asymmetric
                # limits. Choose the legal representation nearest home.
                turns_lo = np.ceil((lo - q - tol) / (2 * np.pi))
                turns_hi = np.floor((hi - q + tol) / (2 * np.pi))
                if np.any(turns_lo > turns_hi):
                    continue
                turns = np.clip(np.rint((Q_DEFAULT - q) / (2 * np.pi)),
                                turns_lo, turns_hi)
                return np.clip(q + 2 * np.pi * turns, lo, hi)
    return None


# ======================================================================
# IK - Numerical Method (bounded Gauss-Newton)
# ======================================================================

def IK_numerical(dh_params, pose, q0=None, joint_limits=None, w_rot=200.0):
    """
    Inverse kinematics for the Lite 6 by bounded Gauss-Newton least squares.

    Solves  min_q ||W e(q)||^2  s.t.  lb <= q <= ub, where e is the pose error
    between FK_dh(q) and the requested pose.

    pose:         [x, y, z, roll, pitch, yaw]  (mm, radians) - same as IK_geometric.
    q0:           starting guess in radians. None seeds from Q_DEFAULT (the home
                  pose) rather than the arm's current angles, so repeated calls
                  are reproducible.
    joint_limits: (6, 2) array of [lo, hi] in radians; defaults to JOINT_LIMITS.
    w_rot:        mm per radian, putting the position and rotation blocks of the
                  residual on one scale. Only matters when a limit binds and the
                  pose cannot be reached exactly.

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.
    """
    pose = np.asarray(pose, dtype=float)
    target_rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    limits = JOINT_LIMITS if joint_limits is None else np.asarray(joint_limits)
    lo, hi = limits.T
    q = np.clip(Q_DEFAULT if q0 is None else q0, lo, hi).astype(float)

    def residual(angles):
        T = FK_dh(dh_params, angles, 6)
        position_error = T[:3, 3] - pose[:3]
        rotation_error = Rotation.from_matrix(
            T[:3, :3] @ target_rotation.T
        ).as_rotvec()
        return np.concatenate((position_error, w_rot * rotation_error))

    epsilon = 1e-6
    for _ in range(200):
        error = residual(q)
        if np.linalg.norm(error) < 1e-3:
            return q

        J = np.zeros((6, 6))
        for i in range(6):
            perturbed = q.copy()
            perturbed[i] += epsilon
            J[:, i] = (residual(perturbed) - error) / epsilon

        # Damping and a small step limit help near singular configurations.
        step = np.linalg.solve(J.T @ J + 1e-2 * np.eye(6), -J.T @ error)
        q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)

    if np.linalg.norm(residual(q)) < 1e-3:
        return q
    return None


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == '__main__':
    # get_transform_from_dh with all zeros should return identity
    T = get_transform_from_dh(0, 0, 0, 0, 0)
    if T is not None:
        print("get_transform_from_dh(all zeros):")
        print(T)
    else:
        print("get_transform_from_dh not yet implemented")
