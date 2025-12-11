import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import casadi as csd
import excavatorConstants as C
import excavatorModel as mod
from excavatorModel import DutyCycle
from NLPSolver import NLP, Mode, Ts, integrator

# ============ 2D 模型里用到的 “几何长度” ============
# 这里改成用 URDF 拟合出来的 L1,L2,L3
LEN_BA = float(C.L1)   # base → lift_boom
LEN_AL = float(C.L2)   # lift_boom → tilt_boom
LEN_LM = float(C.L3)   # tilt_boom → tool_body/gripper tip
TOTAL_LEN = LEN_BA + LEN_AL + LEN_LM


def motor_commands(x, u):
    """
    与 MPCSimulation_modify.py 中 motorCommands 同思路：
    在一个 Ts 内，用小步长 TMotor 对 NLPSolver 的简化积分器细分积分，
    然后再通过 excavatorModel 的函数把电机角速度转换成关节速度。

    现在的状态/控制结构为 4-DOF：
        x = [yaw, q1, q2, q3, yawDot, qDot1, qDot2, qDot3]^T
        u = [yawDDot, qDDot1, qDDot2, qDDot3]^T
    """
    TMotor = 0.001
    NMotor = int(Ts / TMotor)

    # 当前位置下的“机械臂关节”部分（忽略 yaw）
    q_joints = x[1:4]

    # 当前位置下的油缸长度
    actuatorLenPrev = mod.actuatorLen(q_joints)

    deltaActuatorLen = 0
    motorSpdFinal = 0

    xInterpolate = x
    for i in range(NMotor):
        tau = (i + 1) * TMotor
        # 使用 NLPSolver 中的一般 n-DOF integrator 作为“植物”
        xInterpolate = integrator(x, u, tau)

        # 取出其中的 3 个机械关节，用于电机速度映射
        q_full = xInterpolate[0:4]
        qDot_full = xInterpolate[4:8]
        q_j = q_full[1:4]
        qDot_j = qDot_full[1:4]

        motorSpd = mod.motorVel(q_j, qDot_j)
        deltaMotorAng = TMotor * motorSpd
        deltaActuatorLen += deltaMotorAng / 2444.16

        if i == NMotor - 1:
            motorSpdFinal = motorSpd

    # 由执行器长度变化 -> 新的 3 DOF 关节角/角速度
    actuatorLenNew = actuatorLenPrev + deltaActuatorLen
    actuatorVelNew = motorSpdFinal / 2444.16
    qNew = mod.jointAngles(actuatorLenNew)
    qDotNew = mod.jointVel(qNew, actuatorVelNew)

    # yaw 分量沿用 integrator 在 Ts 处的结果
    yaw_next = xInterpolate[0]
    yawDot_next = xInterpolate[4]

    q_full_new = csd.vertcat(yaw_next, qNew)
    qDot_full_new = csd.vertcat(yawDot_next, qDotNew)

    return csd.vertcat(q_full_new, qDot_full_new)


class ExcavatorRealtimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Excavator MPC – Realtime 3D Simulation")

        # ========== 3D 画布 ==========
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ========== 控件区域 ==========
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(
            controls, text="▶ Start (Realtime MPC)", command=self.start_simulation
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            controls, text="⏸ Stop", command=self.stop_simulation
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(
            controls, text="⟲ Reset", command=self.reset_simulation
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_label = ttk.Label(controls, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # ------ 底座 yaw 控制滑条（用于初始 yaw）------
        self.yaw_var = tk.DoubleVar()
        self.yaw_var.set(0.0)  # 初始 0 度
        yaw_scale = tk.Scale(
            controls,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.yaw_var,
            label="Base yaw init [deg]",
            length=180,
        )
        yaw_scale.pack(side=tk.LEFT, padx=5)

        # ------ 外力输入（可调）------
        self.extF = 1000.0        # LIFT 模式的外力（用于可视化箭头长度）
        self.extF_var = tk.DoubleVar(value=self.extF)
        ttk.Label(controls, text="ExtF [N]:").pack(side=tk.LEFT)
        ext_entry = ttk.Entry(controls, textvariable=self.extF_var, width=7)
        ext_entry.pack(side=tk.LEFT, padx=2)

        apply_btn = ttk.Button(controls, text="↺ Apply inputs", command=self.apply_inputs)
        apply_btn.pack(side=tk.LEFT, padx=5)

        # ========== MPC / 仿真状态 ==========
        self.mode = Mode.LIFT
        self.dutyCycle = DutyCycle.S2_30

        self.TSim = 5.0
        self.NSim = int(self.TSim / Ts)

        # 目标末端位姿：仍然是 2D [x, z, phi]
        self.poseDesired = csd.vertcat(0.6, C.yGround, -0.8)
        self.qDesired = mod.inverseKinematics(self.poseDesired)

        # 初始状态：
        # x = [yaw, q1, q2, q3, yawDot, qDot1, qDot2, qDot3]
        yaw0 = math.radians(float(self.yaw_var.get()))
        self.x0 = csd.vertcat(
            yaw0,
            0.5, -0.8, -0.6,   # 你可以根据 URDF 合理调一调初始姿态
            0.0, 0.0, 0.0, 0.0
        )

        self.running = False
        self.k = 0
        self.x = self.x0

        # 末端轨迹记录
        self.tip_traj = []

        # ========== 状态 / I-O 显示面板 ==========
        info = ttk.LabelFrame(self.root, text="State / I-O Monitor")
        info.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.t_var = tk.StringVar(value="0.00")
        self.k_var = tk.StringVar(value="0")
        self.yaw_state_deg_var = tk.StringVar(value="0.0")
        self.q1_deg_var = tk.StringVar(value="0.0")
        self.q2_deg_var = tk.StringVar(value="0.0")
        self.q3_deg_var = tk.StringVar(value="0.0")
        self.tip_x_var = tk.StringVar(value="0.00")
        self.tip_y_var = tk.StringVar(value="0.00")
        self.tip_z_var = tk.StringVar(value="0.00")
        self.extF_state_var = tk.StringVar(value=f"{self.extF:.1f}")

        # Row 0: time / step
        ttk.Label(info, text="t [s]:").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.t_var, width=7).grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="k:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Label(info, textvariable=self.k_var, width=5).grid(row=0, column=3, sticky="w")

        # Row 1: yaw & init yaw
        ttk.Label(info, text="yaw_state [deg]:").grid(row=1, column=0, sticky="w")
        ttk.Label(info, textvariable=self.yaw_state_deg_var, width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(info, text="yaw_init [deg]:").grid(row=1, column=2, sticky="w", padx=(10, 0))
        ttk.Label(info, textvariable=self.yaw_var, width=7).grid(row=1, column=3, sticky="w")

        # Row 2: joint angles
        ttk.Label(info, text="q1,q2,q3 [deg]:").grid(row=2, column=0, sticky="w")
        ttk.Label(info, textvariable=self.q1_deg_var, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(info, textvariable=self.q2_deg_var, width=6).grid(row=2, column=2, sticky="w")
        ttk.Label(info, textvariable=self.q3_deg_var, width=6).grid(row=2, column=3, sticky="w")

        # Row 3: tip position
        ttk.Label(info, text="tip (X,Y,Z) [m]:").grid(row=3, column=0, sticky="w")
        ttk.Label(info, textvariable=self.tip_x_var, width=6).grid(row=3, column=1, sticky="w")
        ttk.Label(info, textvariable=self.tip_y_var, width=6).grid(row=3, column=2, sticky="w")
        ttk.Label(info, textvariable=self.tip_z_var, width=6).grid(row=3, column=3, sticky="w")

        # Row 4: external force
        ttk.Label(info, text="ExtF [N]:").grid(row=4, column=0, sticky="w")
        ttk.Label(info, textvariable=self.extF_state_var, width=7).grid(row=4, column=1, sticky="w")

        # 初始画一帧
        x0_np = np.array(self.x, dtype=float).reshape(-1)
        self.draw_frame(x0_np, t=0.0)

    # ---------- 应用输入参数（外力等） ----------
    def apply_inputs(self):
        try:
            self.extF = float(self.extF_var.get())
            self.extF_state_var.set(f"{self.extF:.1f}")
            self.status_var.set(f"Inputs applied: ExtF={self.extF:.1f} N")
        except Exception:
            self.status_var.set("Failed to parse inputs.")

    # ---------- 运动学：计算关节点 3D 坐标 ----------
    def link_positions_3d(self, yaw, q):
        """
        用 2D 模型 [boom, arm, bucket] 在 x–z 平面上的几何，
        然后绕 Z 轴旋转 yaw 放到 3D 世界坐标中显示。
        """
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])

        # cabin 坐标系下（Y=0）的几何（已经是 URDF 的 L1/L2/L3）
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = p0 + np.array(
            [LEN_BA * math.cos(alpha), 0.0, LEN_BA * math.sin(alpha)]
        )
        p2 = p1 + np.array(
            [LEN_AL * math.cos(alpha + beta), 0.0,
             LEN_AL * math.sin(alpha + beta)]
        )
        p3 = p2 + np.array(
            [LEN_LM * math.cos(alpha + beta + gamma), 0.0,
             LEN_LM * math.sin(alpha + beta + gamma)]
        )

        pts = np.vstack([p0, p1, p2, p3])

        # 绕世界 Z 轴旋转：底座 yaw（来自状态）
        yaw_rad = float(yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        R = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ])

        pts_rot = pts @ R.T
        return pts_rot

    # ---------- 画一帧 ----------
    def draw_frame(self, x_vec, t):
        x_arr = np.array(x_vec, dtype=float).reshape(-1)
        yaw = x_arr[0]
        q = x_arr[1:4]

        pts = self.link_positions_3d(yaw, q)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

        # 记录末端轨迹
        tip = pts[-1].copy()
        self.tip_traj.append(tip)

        # ==== 更新右侧参数显示 ====
        self.t_var.set(f"{t:.2f}")
        self.k_var.set(str(self.k))
        self.yaw_state_deg_var.set(f"{math.degrees(yaw):.1f}")
        self.q1_deg_var.set(f"{math.degrees(q[0]):.1f}")
        self.q2_deg_var.set(f"{math.degrees(q[1]):.1f}")
        self.q3_deg_var.set(f"{math.degrees(q[2]):.1f}")
        self.tip_x_var.set(f"{tip[0]:.2f}")
        self.tip_y_var.set(f"{tip[1]:.2f}")
        self.tip_z_var.set(f"{tip[2]:.2f}")

        self.ax.cla()

        # 挖机连杆
        self.ax.plot(X, Y, Z, "-o", label="excavator arm")

        # 末端轨迹（虚线）
        if len(self.tip_traj) > 1:
            traj = np.vstack(self.tip_traj)
            self.ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                "--", linewidth=1, label="tip trajectory"
            )

        # 外力方向箭头（在 tip 上）
        force_dir = None
        if self.mode == Mode.LIFT:
            force_dir = np.array([0.0, 0.0, -1.0])
        elif self.mode == Mode.DIG and len(pts) >= 2:
            last_seg = pts[-1] - pts[-2]
            norm = np.linalg.norm(last_seg)
            if norm > 1e-6:
                force_dir = -last_seg / norm

        if force_dir is not None and abs(self.extF) > 1e-6:
            arrow_len = 0.002 * abs(self.extF)
            self.ax.quiver(
                tip[0], tip[1], tip[2],
                force_dir[0], force_dir[1], force_dir[2],
                length=arrow_len,
                normalize=True,
                label="external force",
            )

        L = TOTAL_LEN + 0.5
        self.ax.plot(
            [-L, L],
            [0, 0],
            [float(C.yGround), float(C.yGround)],
            "k--",
            linewidth=1,
            label="ground",
        )

        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_zlim(float(C.yGround) - 0.2, L)

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title(f"Realtime MPC – t = {t:.2f} s  (k = {self.k})")

        self.ax.view_init(elev=25, azim=-60)
        self.ax.legend(loc="upper left", fontsize=8)

        self.canvas.draw()

    # ---------- 单步 MPC 更新 ----------
    def step_mpc(self):
        if self.k >= self.NSim:
            self.status_var.set(f"Finished: k={self.k} >= NSim={self.NSim}")
            return False

        try:
            opti = NLP(self.mode, self.extF, self.dutyCycle)
            sol = opti.solveNLP(self.x, self.poseDesired)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"NLP solve failed at k={self.k}: {e}")
            return False

        try:
            u0 = sol.value(opti.u[:, 0])  # (4,)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Extract u0 failed at k={self.k}: {e}")
            return False

        u0_dm = csd.DM(u0)

        try:
            self.x = motor_commands(sol.value(opti.x[:, 0]), u0_dm)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"motor_commands failed at k={self.k}: {e}")
            return False

        self.k += 1
        return True

    # ---------- 动画循环（实时仿真） ----------
    def update_frame(self):
        if not self.running:
            return

        ok = self.step_mpc()
        if not ok:
            self.running = False
            self.start_button.config(text="▶ Start (Realtime MPC)")
            return

        t = self.k * Ts
        x_np = np.array(self.x, dtype=float).reshape(-1)
        self.draw_frame(x_np, t)

        self.root.after(10, self.update_frame)

    # ---------- 控件回调 ----------
    def start_simulation(self):
        if self.running:
            return
        self.status_var.set("Running realtime MPC...")
        self.running = True
        self.start_button.config(text="▶ Running...")
        self.update_frame()

    def stop_simulation(self):
        if not self.running:
            return
        self.running = False
        self.status_var.set("Paused.")
        self.start_button.config(text="▶ Start (Realtime MPC)")

    def reset_simulation(self):
        if self.running:
            self.stop_simulation()
        self.k = 0

        # 根据当前滑条重置初始 yaw
        yaw0 = math.radians(float(self.yaw_var.get()))
        self.x0 = csd.vertcat(
            yaw0,
            0.5, -0.8, -0.6,
            0.0, 0.0, 0.0, 0.0
        )
        self.x = self.x0

        # 重置轨迹
        self.tip_traj = []

        x0_np = np.array(self.x, dtype=float).reshape(-1)
        self.draw_frame(x0_np, t=0.0)
        self.status_var.set("Reset to initial state.")


def main():
    root = tk.Tk()
    app = ExcavatorRealtimeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
