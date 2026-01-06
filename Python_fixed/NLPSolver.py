import casadi as csd
from enum import Enum
import numpy as np

import excavatorModel as mod
from excavatorModel import DutyCycle

# Prediction horizon (seconds) and steps
T = 2.0
N = 20
Ts = T / N


def integrator(x, u, dt=Ts):
    """Simple 2nd-order joint-space integrator for an n-DOF system.

    State x  = [q0, ..., q_{n-1}, qDot0, ..., qDot_{n-1}]^T   (2n,)
    Control u = [qDDot0, ..., qDDot_{n-1}]^T                   (n,)

    q_next    = q + qDot*dt + 0.5*qDDot*dt^2
    qDot_next = qDot + qDDot*dt
    """
    if isinstance(x, (list, tuple)):
        x = csd.vertcat(*x)
    if isinstance(u, (list, tuple)):
        u = csd.vertcat(*u)

    try:
        n = int(u.size1())
    except AttributeError:
        n = int(len(u))

    q = x[0:n]
    qDot = x[n:2 * n]
    qDDot = u

    q_next = q + qDot * dt + 0.5 * qDDot * dt ** 2
    qDot_next = qDot + qDDot * dt

    return csd.vertcat(q_next, qDot_next)


class Mode(Enum):
    NO_LOAD = 0
    LIFT = 1
    DIG = 2


def _to_np_1d(x, n, name="x"):
    """Convert list/tuple/np/casadi DM/MX to numpy float (n,)."""
    try:
        # CasADi DM/MX/SX etc.
        if "casadi" in str(type(x)).lower():
            x = np.array(csd.DM(x).full()).reshape(-1)
        else:
            x = np.asarray(x, dtype=float).reshape(-1)
    except Exception as e:
        raise ValueError(f"Failed to convert {name} to numeric: {e}. Value={x}")

    if x.size != n:
        raise ValueError(f"{name} must have {n} elements, got {x.size}. Value={x}")
    return x.astype(float)


def _is_finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


class NLP:
    """Nonlinear MPC for a 4-DOF excavator arm:
        q = [yaw, alpha, beta, gamma]^T

    Dynamics: joint-space double integrator.
    Output for cost: planar EE pose from excavatorModel.forwardKinematics(q[1:4]).

    IMPORTANT FIXES:
    - Add basic constraints to prevent runaway accelerations/velocities.
    - Guard Opti.set_value against NaN/Inf (CasADi 'v.is_regular()' assertion).
    """

    def __init__(self, mode: Mode = Mode.NO_LOAD,
                 ext_force: float = 0.0,
                 duty_cycle: float = 1.0):
        self.mode = mode
        self.ext_force = float(ext_force)
        if isinstance(duty_cycle, DutyCycle):
            self.duty_cycle = float(duty_cycle.value)
        else:
            self.duty_cycle = float(duty_cycle)

        # 4 DOF: yaw + 3 joints
        self.ndof = 4
        self.nx = 2 * self.ndof
        self.nu = self.ndof

        self.opti = csd.Opti()
        self.x = self.opti.variable(self.nx, N + 1)
        self.u = self.opti.variable(self.nu, N)

        # Parameters
        self.x0 = self.opti.parameter(self.nx)       # (8,1)
        # World-frame target: [x_w, y_w, z_w], and desired tool angle phi (rad)
        self.targetXYZ = self.opti.parameter(3)      # (3,1)
        self.targetPhi = self.opti.parameter(1)      # (1,1)

        # Cost weights
        # Position/orientation tracking weights (world frame TCP)
        self.Q_pos = np.diag([300.0, 300.0, 300.0])
        self.Q_phi = 50.0
        # Penalize velocities lightly
        self.Q_vel = 0.1 * np.eye(self.ndof)
        # Control effort / rate (lighter to allow faster convergence)
        self.R = 0.005 * np.eye(self.ndof)
        self.Rd = 0.05 * np.eye(self.ndof)

        # ===== Constraints (keep solver stable) =====
        # yaw is free in principle, but bounding helps numerical stability
        self.yaw_min = -np.pi
        self.yaw_max =  np.pi

        # Joint limits from your URDF
        self.q_min = np.array([0.0, -0.5235988, -2.0943951], dtype=float)
        self.q_max = np.array([1.3962634,  1.3962634,  0.0], dtype=float)

        # Velocity limits (tunable)
        self.qdot_min = -2.0 * np.ones(3, dtype=float)
        self.qdot_max =  2.0 * np.ones(3, dtype=float)
        self.yawdot_min = -2.0
        self.yawdot_max =  2.0

        # Acceleration limits (tunable)
        # Keep yaw a bit smaller to avoid spinning explosions
        self.uddot_min = np.array([-2.0, -4.0, -4.0, -4.0], dtype=float)
        self.uddot_max = np.array([ 2.0,  4.0,  4.0,  4.0], dtype=float)

        self._build_ocp()

        self.initial_guess_x = np.zeros((self.nx, N + 1), dtype=float)
        self.initial_guess_u = np.zeros((self.nu, N), dtype=float)

        # last-good guards for CasADi is_regular()
        self._last_x_current = np.zeros((self.nx,), dtype=float)
        self._last_pose_des = np.zeros((3,), dtype=float)
        self._last_pose_phi = 0.0

    def _build_ocp(self):
        J = 0
        previous_u = csd.MX.zeros(self.nu, 1)

        self.opti.subject_to(self.x[:, 0] == self.x0)

        # Pre-cast bounds for Opti
        umin = csd.DM(self.uddot_min).reshape((self.nu, 1))
        umax = csd.DM(self.uddot_max).reshape((self.nu, 1))

        for k in range(N):
            xk = self.x[:, k]
            uk = self.u[:, k]

            q_full = xk[0:self.ndof]
            qDot_full = xk[self.ndof:2 * self.ndof]

            # ===== box constraints =====
            # control
            self.opti.subject_to(umin <= uk)
            self.opti.subject_to(uk <= umax)

            # yaw bounds
            self.opti.subject_to(self.yaw_min <= q_full[0])
            self.opti.subject_to(q_full[0] <= self.yaw_max)
            self.opti.subject_to(self.yawdot_min <= qDot_full[0])
            self.opti.subject_to(qDot_full[0] <= self.yawdot_max)

            # joint bounds
            self.opti.subject_to(self.q_min[0] <= q_full[1])
            self.opti.subject_to(q_full[1] <= self.q_max[0])
            self.opti.subject_to(self.q_min[1] <= q_full[2])
            self.opti.subject_to(q_full[2] <= self.q_max[1])
            self.opti.subject_to(self.q_min[2] <= q_full[3])
            self.opti.subject_to(q_full[3] <= self.q_max[2])

            self.opti.subject_to(self.qdot_min[0] <= qDot_full[1])
            self.opti.subject_to(qDot_full[1] <= self.qdot_max[0])
            self.opti.subject_to(self.qdot_min[1] <= qDot_full[2])
            self.opti.subject_to(qDot_full[2] <= self.qdot_max[1])
            self.opti.subject_to(self.qdot_min[2] <= qDot_full[3])
            self.opti.subject_to(qDot_full[3] <= self.qdot_max[2])

            # ===== cost =====
            q_joints = q_full[1:4]
            pose_k = mod.forwardKinematics(q_joints)  # [x_c, z_c, phi] in cabin frame

            # world-frame TCP position
            yaw_k = q_full[0]
            xw = pose_k[0] * csd.cos(yaw_k)
            yw = pose_k[0] * csd.sin(yaw_k)
            zw = pose_k[1]
            pos_w = csd.vertcat(xw, yw, zw)

            e_pos = pos_w - self.targetXYZ
            e_phi = pose_k[2] - self.targetPhi

            J += (
                csd.mtimes([e_pos.T, self.Q_pos, e_pos])
                + self.Q_phi * e_phi ** 2
                + csd.mtimes([qDot_full.T, self.Q_vel, qDot_full])
                + csd.mtimes([uk.T, self.R, uk])
                + csd.mtimes([(uk - previous_u).T, self.Rd, (uk - previous_u)])
            )

            x_next = integrator(xk, uk, Ts)
            self.opti.subject_to(self.x[:, k + 1] == x_next)

            previous_u = uk

        # terminal cost
        xN = self.x[:, N]
        q_full_N = xN[0:self.ndof]
        qDot_full_N = xN[self.ndof:2 * self.ndof]

        q_joints_N = q_full_N[1:4]
        pose_N = mod.forwardKinematics(q_joints_N)

        yaw_N = q_full_N[0]
        xwN = pose_N[0] * csd.cos(yaw_N)
        ywN = pose_N[0] * csd.sin(yaw_N)
        zwN = pose_N[1]
        pos_wN = csd.vertcat(xwN, ywN, zwN)

        e_posN = pos_wN - self.targetXYZ
        e_phiN = pose_N[2] - self.targetPhi

        J += (
            csd.mtimes([e_posN.T, self.Q_pos, e_posN])
            + self.Q_phi * e_phiN ** 2
            + csd.mtimes([qDot_full_N.T, self.Q_vel, qDot_full_N])
        )

        self.opti.minimize(J)

        ipopt_opts = {
            "max_iter": 150,
            "print_level": 0,
        }
        opts = {
            "ipopt": ipopt_opts,
            "print_time": 0,
        }
        self.opti.solver("ipopt", opts)

    def solveNLP(self, x_current, target_xyz, target_phi=0.0):
        """Solve the MPC NLP with world-frame target (x, y, z, phi).

        Fix: Never allow NaN/Inf into Opti.set_value() to avoid CasADi
        assertion "v.is_regular()".
        """
        # convert
        try:
            x_np = _to_np_1d(x_current, self.nx, "x_current")
        except Exception:
            x_np = None

        try:
            p_np = _to_np_1d(target_xyz, 3, "target_xyz")
        except Exception:
            p_np = None

        try:
            phi_np = float(target_phi)
        except Exception:
            phi_np = None

        # finite guard + fallback
        if x_np is None or not _is_finite(x_np):
            x_np = self._last_x_current.copy()
        else:
            self._last_x_current = x_np.copy()

        # ------------------------------------------------------------------
        # Feasibility guard:
        # Ipopt will report "Infeasible_Problem_Detected" if the *current state*
        # violates the box constraints, because we enforce x[:,0] == x0 and also
        # enforce bounds on x[:,k] for all k (including k=0).
        #
        # In your URDF, revolute_tilt has lower bound -0.5236 rad, but older
        # MATLAB defaults used -0.8 rad. If we pass -0.8 into x0, the NLP becomes
        # infeasible and the GUI will appear "stuck".
        #
        # So we project x_current into the constraint box before set_value.
        # ------------------------------------------------------------------
        # yaw
        x_np[0] = float(np.clip(x_np[0], self.yaw_min, self.yaw_max))
        # joints
        x_np[1] = float(np.clip(x_np[1], self.q_min[0], self.q_max[0]))
        x_np[2] = float(np.clip(x_np[2], self.q_min[1], self.q_max[1]))
        x_np[3] = float(np.clip(x_np[3], self.q_min[2], self.q_max[2]))
        # velocities
        x_np[4] = float(np.clip(x_np[4], self.yawdot_min, self.yawdot_max))
        x_np[5] = float(np.clip(x_np[5], self.qdot_min[0], self.qdot_max[0]))
        x_np[6] = float(np.clip(x_np[6], self.qdot_min[1], self.qdot_max[1]))
        x_np[7] = float(np.clip(x_np[7], self.qdot_min[2], self.qdot_max[2]))

        if p_np is None or not _is_finite(p_np):
            p_np = self._last_pose_des.copy()
        else:
            self._last_pose_des = p_np.copy()

        if phi_np is None or not np.isfinite(phi_np):
            phi_np = self._last_pose_phi
        else:
            self._last_pose_phi = float(phi_np)

        # set values as DM column vectors
        self.opti.set_value(self.x0, csd.DM(x_np).reshape((self.nx, 1)))
        self.opti.set_value(self.targetXYZ, csd.DM(p_np).reshape((3, 1)))
        self.opti.set_value(self.targetPhi, csd.DM([[phi_np]]))

        self.opti.set_initial(self.x, self.initial_guess_x)
        self.opti.set_initial(self.u, self.initial_guess_u)

        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            print("=== NLP solve failed, using debug solution ===")
            print(e)
            sol = self.opti.debug

        X_opt = np.array(sol.value(self.x))
        U_opt = np.array(sol.value(self.u))

        # shift warm-start
        X_shift = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_shift = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])

        self.initial_guess_x = X_shift
        self.initial_guess_u = U_shift

        return sol
