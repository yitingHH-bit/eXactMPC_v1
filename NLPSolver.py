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
    # Make sure we work with CasADi types internally when needed
    if isinstance(x, (list, tuple)):
        x = csd.vertcat(*x)
    if isinstance(u, (list, tuple)):
        u = csd.vertcat(*u)

    # Determine number of DOFs from control dimension
    try:
        n = int(u.size1())
    except AttributeError:
        # Fallback for numpy arrays / python lists
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
    Nonlinear MPC for the 3-link excavator arm (planar dynamics)
    with end-effector position/orientation tracking in the x-z plane.

    The base rotation/yaw is handled at the kinematics/visualisation
    level and is not yet part of the dynamic state here.
    """

    def __init__(self, mode: Mode = Mode.NO_LOAD,
                 ext_force: float = 0.0,
                 duty_cycle: float = 1.0):
        # These are kept for compatibility with your GUI,
        # currently not used directly in the cost.
        self.mode = mode
        self.ext_force = float(ext_force)
        if isinstance(duty_cycle, DutyCycle):
            self.duty_cycle = float(duty_cycle.value)
        else:
            self.duty_cycle = float(duty_cycle)

        
        
        # Degrees of freedom:
        #   0: base yaw
        #   1-3: boom/arm/bucket joint angles
        self.ndof = 4

        # State:  x = [q0..q3, qDot0..qDot3]^T  (8,)
        # Control: u = [qDDot0..qDDot3]^T       (4,)
        self.nx = 2 * self.ndof
        self.nu = self.ndof

        # Optimisation object
        self.opti = csd.Opti()

        # Decision variables
        self.x = self.opti.variable(self.nx, N + 1)   # state trajectory
        self.u = self.opti.variable(self.nu, N)       # control trajectory

        # Parameters: initial state and desired end-effector pose [x,z,phi]
        self.x0 = self.opti.parameter(self.nx)
        self.poseDesired = self.opti.parameter(3)

        # Cost weights


        # Cost weights
        self.Q_pos = np.diag([80.0, 80.0])     # position [x,z]
        self.Q_phi = 20.0                      # orientation about end-effector
        self.Q_vel = 0.1 * np.eye(self.ndof)   # joint & yaw velocities
        self.R = 0.01 * np.eye(self.ndof)      # control effort
        self.Rd = 0.1 * np.eye(self.ndof)      # move suppression Δu

        # Simple joint limits (rad) – 可以根据你真实模型调整
        # 先用比较宽的关节范围，保证 x0 在可行域里  
        # 以后你可以根据真实机械结构再收紧  
        self.q_min = np.array([-3.14, -3.14, -3.14])
        self.q_max = np.array([ 3.14,  3.14,  3.14])

        # Velocity & acceleration limits
        self.qDot_min = -2.0 * np.ones(3)
        self.qDot_max =  2.0 * np.ones(3)
        self.u_min = -4.0 * np.ones(3)
        self.u_max =  4.0 * np.ones(3)

        # Build the optimal control problem
        self._build_ocp()

        # Warm-start storage
        self.initial_guess_x = np.zeros((self.nx, N + 1))
        self.initial_guess_u = np.zeros((self.nu, N))

    # ------------------------------------------------------------------
    # OCP construction
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # OCP construction（暂时不加任何上下界约束，先保证问题可行）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # OCP construction (now with 4-DOF state including base yaw)
    # ------------------------------------------------------------------
    def _build_ocp(self):
        J = 0
        previous_u = csd.MX.zeros(self.nu, 1)

        # Initial state constraint
        self.opti.subject_to(self.x[:, 0] == self.x0)

        for k in range(N):
            xk = self.x[:, k]
            uk = self.u[:, k]

            # Full generalized coordinates/velocities (yaw + 3 joints)
            q_full = xk[0:self.ndof]
            qDot_full = xk[self.ndof:2 * self.ndof]

            # For planar kinematics we only use the arm joints (skip yaw)
            q_joints = q_full[1:4]
            qDot_joints = qDot_full[1:4]

            # Planar forward kinematics from current joint angles
            # excavatorModel.forwardKinematics(q) must return [x,z,phi]
            pose_k = mod.forwardKinematics(q_joints)

            # Tracking error in the boom plane
            e_pos = pose_k[0:2] - self.poseDesired[0:2]
            e_phi = pose_k[2] - self.poseDesired[2]

            # Stage cost:
            #  - end-effector position/orientation tracking
            #  - penalise ALL velocities (including yawDot)
            #  - penalise ALL accelerations (including yawDDot)
            #  - penalise control variation Δu
            J += (
                csd.mtimes([e_pos.T, self.Q_pos, e_pos])
                + self.Q_phi * e_phi ** 2
                + csd.mtimes([qDot_full.T, self.Q_vel, qDot_full])
                + csd.mtimes([uk.T, self.R, uk])
                + csd.mtimes([(uk - previous_u).T, self.Rd, (uk - previous_u)])
            )

            # Simple double-integrator dynamics in joint space (4 DOF)
            x_next = integrator(xk, uk, Ts)
            self.opti.subject_to(self.x[:, k + 1] == x_next)

            previous_u = uk

        # Terminal cost – same structure as stage cost but without Δu term
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

        # Solver options
        ipopt_opts = {
            "max_iter": 80,
            "print_level": 0,
        }
        opts = {
            "ipopt": ipopt_opts,
            "print_time": 0,
        }
        self.opti.solver("ipopt", opts)

    # ------------------------------------------------------------------
    # Solve the MPC problem for current state & desired pose
    # ------------------------------------------------------------------
    def solveNLP(self, x_current, pose_desired):
        """
        x_current: array-like (6,) – current [q1,q2,q3,qDot1,qDot2,qDot3]
        pose_desired: array-like (3,) – desired [x,z,phi]
        """
        x_current = np.asarray(x_current, dtype=float).reshape(self.nx)
        pose_desired = np.asarray(pose_desired, dtype=float).reshape(3)

        # Update parameters
        self.opti.set_value(self.x0, x_current)
        self.opti.set_value(self.poseDesired, pose_desired)

        # Warm start
        self.opti.set_initial(self.x, self.initial_guess_x)
        self.opti.set_initial(self.u, self.initial_guess_u)

        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            print("=== NLP solve failed, using debug solution ===")
            print(e)
            sol = self.opti.debug

        # Extract optimal trajectories
        X_opt = np.array(sol.value(self.x))
        U_opt = np.array(sol.value(self.u))

        # Shift for next warm start (receding horizon)
        X_shift = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_shift = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])

        self.initial_guess_x = X_shift
        self.initial_guess_u = U_shift

        return sol
