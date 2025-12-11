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
    Simple second-order joint-space integrator.

    State x  = [q1, q2, q3, qDot1, qDot2, qDot3]^T   (6,)
    Control u = [qDDot1, qDDot2, qDDot3]^T           (3,)

    q_next    = q + qDot*dt + 0.5*qDDot*dt^2
    qDot_next = qDot + qDDot*dt
    """
    # Make sure we work with CasADi types internally when needed
    if isinstance(x, (list, tuple)):
        x = csd.vertcat(*x)
    if isinstance(u, (list, tuple)):
        u = csd.vertcat(*u)

    q = x[0:3]
    qDot = x[3:6]
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

        # Dimensions
        self.nx = 6   # [q1,q2,q3,qDot1,qDot2,qDot3]
        self.nu = 3   # joint accelerations qDDot

        # Optimisation object
        self.opti = csd.Opti()

        # Decision variables
        self.x = self.opti.variable(self.nx, N + 1)   # state trajectory
        self.u = self.opti.variable(self.nu, N)       # control trajectory

        # Parameters: initial state and desired end-effector pose [x,z,phi]
        self.x0 = self.opti.parameter(self.nx)
        self.poseDesired = self.opti.parameter(3)

        # Cost weights
        self.Q_pos = np.diag([80.0, 80.0])     # position [x,z]
        self.Q_phi = 20.0                      # orientation
        self.Q_vel = 0.1 * np.eye(3)           # joint velocities
        self.R = 0.01 * np.eye(3)              # control effort
        self.Rd = 0.1 * np.eye(3)              # move suppression Δu

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
    def _build_ocp(self):
        J = 0
        # 上一个控制输入，用于 Δu 惩罚项
        previous_u = csd.MX.zeros(self.nu, 1)

        # 初始状态约束：x(0) = x0
        self.opti.subject_to(self.x[:, 0] == self.x0)

        # --- 阶段代价 + 动力学约束 ---
        for k in range(N):
            xk = self.x[:, k]
            uk = self.u[:, k]
            qk = xk[0:3]
            qDotk = xk[3:6]

            # 正运动学：q -> [x, z, phi]
            pose_k = mod.forwardKinematics(qk)

            # 跟踪误差
            e_pos = pose_k[0:2] - self.poseDesired[0:2]   # 位置 [x,z]
            e_phi = pose_k[2] - self.poseDesired[2]       # 姿态 phi

            # 阶段代价
            J += (
                csd.mtimes([e_pos.T, self.Q_pos, e_pos])      # 末端位置误差
                + self.Q_phi * e_phi ** 2                     # 末端姿态误差
                + csd.mtimes([qDotk.T, self.Q_vel, qDotk])    # 关节速度
                + csd.mtimes([uk.T, self.R, uk])              # 控制能量
                + csd.mtimes([(uk - previous_u).T, self.Rd, (uk - previous_u)])  # 控制变化
            )

            # 简单双积分动力学 x_{k+1} = f(x_k, u_k)
            x_next = integrator(xk, uk)
            self.opti.subject_to(self.x[:, k + 1] == x_next)

            previous_u = uk

        # --- 终端代价（只罚跟踪和速度，不罚 Δu） ---
        xN = self.x[:, N]
        qN = xN[0:3]
        qDotN = xN[3:6]

        pose_N = mod.forwardKinematics(qN)
        e_posN = pose_N[0:2] - self.poseDesired[0:2]
        e_phiN = pose_N[2] - self.poseDesired[2]

        J += (
            csd.mtimes([e_posN.T, self.Q_pos, e_posN])
            + self.Q_phi * e_phiN ** 2
            + csd.mtimes([qDotN.T, self.Q_vel, qDotN])
        )

        # 设置目标函数
        self.opti.minimize(J)

        # Ipopt 配置（可以先保持安静一点）
        ipopt_opts = {
            "max_iter": 80,
            "print_level": 0,         # 0=最安静；你想看详细输出可以改成 5
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "acceptable_obj_change_tol": 1e-3,
        }
        opts = {
            "ipopt": ipopt_opts,
            "print_time": False,
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
