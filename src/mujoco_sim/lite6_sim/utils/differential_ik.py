"""
Velocity (differential) inverse kinematics.

Maps a desired end-effector (6-row) twist to joint velocities by solving a small QP:

    min_qd   || J qd - v_des ||^2_W  +  lambda * || qd ||^2
    s.t.     max(-qd_lim, (q_lo - q)/dt)  <=  qd  <=  min(qd_lim, (q_hi - q)/dt)

The single box constraint carries both the joint speed limit and the joint travel
limit (expressed as the velocity that would just reach it in one tick).

This is a pure jog, so there is a commanded velocity but no pose target.

proxsuite is the intended backend, with damped least squares as a fallback.
"""
import atexit
import weakref

import numpy as np

try:
    import proxsuite
    _PROXSUITE_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _PROXSUITE_AVAILABLE = False

__all__ = ["DifferentialIKSolver"]

# Solvers holding a proxsuite QP, released at exit.
# lcm.LCM holds bound methods of the bridge but has no tp_traverse, so that cycle is
# uncollectable: the bridge and its QP live to interpreter shutdown, where nanobind
# reports a leak. Harmless -- the process is exiting -- but noisy.
_LIVE_SOLVERS = weakref.WeakSet()


@atexit.register
def _release_qp_handles():
    for solver in list(_LIVE_SOLVERS):
        solver.close()

# Task weights.
W_POS = 1.0
W_ROT = 0.1

# Tikhonov term.
DAMPING = 1e-4

DEFAULT_QD_LIMIT = np.pi  # rad/s, per joint


class DifferentialIKSolver:
    """Twist -> joint velocity, subject to joint speed and travel limits."""

    def __init__(self, joint_lo, joint_hi, w_pos=W_POS, w_rot=W_ROT,
                 damping=DAMPING, qd_limit=DEFAULT_QD_LIMIT):
        self.joint_lo = np.asarray(joint_lo, dtype=float).ravel()
        self.joint_hi = np.asarray(joint_hi, dtype=float).ravel()
        if self.joint_lo.shape != self.joint_hi.shape:
            raise ValueError("joint_lo and joint_hi must have the same length")
        self.n = int(self.joint_lo.size)
        self.damping = float(damping)
        self.qd_limit = float(qd_limit)
        self.W = np.diag([w_pos] * 3 + [w_rot] * 3)

        self._qp = None
        if _PROXSUITE_AVAILABLE:
            # n variables, no equalities, n box constraints. Allocated once;
            # solve() only refreshes the numbers.
            self._qp = proxsuite.proxqp.dense.QP(self.n, 0, self.n)
            self._qp.settings.eps_abs = 1e-9
            self._qp.settings.verbose = False
            self._qp.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT)
            self._qp_initialized = False
            _LIVE_SOLVERS.add(self)
        else:
            print("WARNING: proxsuite not found. Differential IK falls back to "
                  "damped least squares (joint limits are clipped, not constrained).")

    def _velocity_bounds(self, q_cmd, dt, qd_limit):
        q_cmd = np.asarray(q_cmd, dtype=float).ravel()
        if q_cmd.size != self.n or not np.all(np.isfinite(q_cmd)):
            raise ValueError(f"q_cmd must be a finite vector of length {self.n}")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")

        limit = self.qd_limit if qd_limit is None else float(qd_limit)
        if not np.isfinite(limit) or limit < 0.0:
            raise ValueError("qd_limit must be non-negative and finite")

        # Clamp only for computing the one-step travel bound. This also gives a
        # feasible recovery direction if numerical drift puts q just outside a
        # joint limit.
        q_bounded = np.clip(q_cmd, self.joint_lo, self.joint_hi)
        lower = np.maximum(-limit, (self.joint_lo - q_bounded) / dt)
        upper = np.minimum( limit, (self.joint_hi - q_bounded) / dt)
        return lower, upper

    def solve(self, jacobian, v_des, q_cmd, dt, qd_limit=None):
        """Return joint velocities (n,) tracking `v_des`, or None if unsolvable.

        jacobian: (6, n) end-effector Jacobian [v; w] = J @ qd, in the same frame as v_des.
        v_des:    (6,) desired twist [vx, vy, vz, wx, wy, wz].
        q_cmd:    (n,) configuration the Jacobian was evaluated at -- the *commanded*
                  one, so the caller integrates its own state rather than chasing
                  measurement noise.
        dt:       control period (s); sets how much travel one tick may consume.
        qd_limit: per-joint speed cap (rad/s); None uses the constructor default.
        """
        J = np.asarray(jacobian, dtype=float)
        v = np.asarray(v_des, dtype=float).ravel()
        if J.shape != (6, self.n) or v.shape != (6,):
            return None
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(v)):
            return None

        try:
            lower, upper = self._velocity_bounds(q_cmd, dt, qd_limit)
        except (TypeError, ValueError):
            return None

        H = J.T @ self.W @ J + self.damping * np.eye(self.n)
        # proxqp uses 1/2*x.T*H*x + g.T*x. The factor of two common
        # to both terms is immaterial to the minimizer.
        g = -(J.T @ self.W @ v)

        if self._qp is not None:
            try:
                C = np.eye(self.n)
                if not self._qp_initialized:
                    self._qp.init(H=H, g=g, C=C, l=lower, u=upper)
                    self._qp_initialized = True
                else:
                    self._qp.update(H=H, g=g, C=C, l=lower, u=upper)
                self._qp.solve()
                result = np.asarray(self._qp.results.x, dtype=float).ravel()
                if result.size == self.n and np.all(np.isfinite(result)):
                    return np.clip(result, lower, upper)
            except Exception:
                # Keep Cartesian jog available if the optional QP backend
                # rejects an update or fails numerically near a singularity.
                pass

        result = self._solve_damped_least_squares(H, g)
        if result is None:
            return None
        return np.clip(result, lower, upper)

    def _solve_damped_least_squares(self, H, g):
        try:
            result = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            try:
                result = np.linalg.lstsq(H, -g, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
        if result.size != self.n or not np.all(np.isfinite(result)):
            return None
        return result

    def close(self):
        """Release the proxsuite QP. Safe to call more than once."""
        self._qp = None
        self._qp_initialized = False

    def clamp_to_limits(self, q):
        """Clamp a joint vector into the travel limits."""
        return np.clip(np.asarray(q, dtype=float).ravel(), self.joint_lo, self.joint_hi)
