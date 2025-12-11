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
    """
    Simple second-order joint-space integrator for an n-DOF system.

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


class NLP:
    """
    Nonlinear MPC for a 4-DOF excavator arm:
        q = [yaw, alpha, beta, gamma]^T
    We still use a simple double-integrator in joint space for dynamics,
    and use excavatorModel.forwardKinematics(q[1:4]) for planar EE pose.
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

        # State:  x = [q0..q3, qDot0..q3]^T  (8,)
        # Control: u = [qDDot0..qDDot3]^T    (4,)
        self.nx = 2 * self.ndof
        self.nu = self.ndof

        self.opti = csd.Opti()
        self.x = self.opti.variable(self.nx, N + 1)
        self.u = self.opti.variable(self.nu, N)

        # Parameters: initial state and desired end-effector pose [x,z,phi]
        self.x0 = self.opti.parameter(self.nx)
        self.poseDesired = self.opti.parameter(3)

        # Cost weights
        self.Q_pos = np.diag([80.0, 80.0])     # position [x,z]
        self.Q_phi = 20.0                      # orientation
        self.Q_vel = 0.1 * np.eye(self.ndof)   # joint & yaw velocities
        self.R = 0.01 * np.eye(self.ndof)      # control effort
        self.Rd = 0.1 * np.eye(self.ndof)      # move suppression Δu

        # （当前没有显式用到这些 joint limit，只保留以备后用）
        self.q_min = np.array([-3.14, -3.14, -3.14])
        self.q_max = np.array([ 3.14,  3.14,  3.14])
        self.qDot_min = -2.0 * np.ones(3)
        self.qDot_max =  2.0 * np.ones(3)
        self.u_min = -4.0 * np.ones(3)
        self.u_max =  4.0 * np.ones(3)

        self._build_ocp()

        self.initial_guess_x = np.zeros((self.nx, N + 1))
        self.initial_guess_u = np.zeros((self.nu, N))

    def _build_ocp(self):
        J = 0
        previous_u = csd.MX.zeros(self.nu, 1)

        self.opti.subject_to(self.x[:, 0] == self.x0)

        for k in range(N):
            xk = self.x[:, k]
            uk = self.u[:, k]

            q_full = xk[0:self.ndof]
            qDot_full = xk[self.ndof:2 * self.ndof]

            q_joints = q_full[1:4]
            pose_k = mod.forwardKinematics(q_joints)

            e_pos = pose_k[0:2] - self.poseDesired[0:2]
            e_phi = pose_k[2] - self.poseDesired[2]

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

        xN = self.x[:, N]
        q_full_N = xN[0:self.ndof]
        qDot_full_N = xN[self.ndof:2 * self.ndof]

        q_joints_N = q_full_N[1:4]
        pose_N = mod.forwardKinematics(q_joints_N)

        e_posN = pose_N[0:2] - self.poseDesired[0:2]
        e_phiN = pose_N[2] - self.poseDesired[2]

        J += (
            csd.mtimes([e_posN.T, self.Q_pos, e_posN])
            + self.Q_phi * e_phiN ** 2
            + csd.mtimes([qDot_full_N.T, self.Q_vel, qDot_full_N])
        )

        self.opti.minimize(J)

        ipopt_opts = {
            "max_iter": 80,
            "print_level": 0,
        }
        opts = {
            "ipopt": ipopt_opts,
            "print_time": 0,
        }
        self.opti.solver("ipopt", opts)

    def solveNLP(self, x_current, pose_desired):
        """
        x_current: array-like (8,) – current [yaw,q1,q2,q3,yawDot,qDot1,qDot2,qDot3]
        pose_desired: array-like (3,) – desired [x,z,phi]
        """
        x_current = np.asarray(x_current, dtype=float).reshape(self.nx)
        pose_desired = np.asarray(pose_desired, dtype=float).reshape(3)

        self.opti.set_value(self.x0, x_current)
        self.opti.set_value(self.poseDesired, pose_desired)

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

        X_shift = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_shift = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])

        self.initial_guess_x = X_shift
        self.initial_guess_u = U_shift

        return sol
