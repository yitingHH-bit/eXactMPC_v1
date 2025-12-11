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
# 还是沿用你原来 eXactMPC 的 2D 参数，只是用于 3D 可视化
LEN_BA = float(C.lenBA)   # base → boom
LEN_AL = float(C.lenAL)   # boom → arm
LEN_LM = float(C.lenLM)   # arm → tip
TOTAL_LEN = LEN_BA + LEN_AL + LEN_LM


def motor_commands(x, u):
    """
    与 MPCSimulation_modify.py 中 motorCommands 同思路：
    在一个 Ts 内，用小步长 TMotor 对 NLPSolver 的简化积分器细分积分，
    然后再通过 excavatorModel 的函数把电机角速度转换成关节速度。
    """
    TMotor = 0.001
    NMotor = int(Ts / TMotor)

    # 当前位置下的油缸长度
    actuatorLenPrev = mod.actuatorLen(x[0:3])

    deltaActuatorLen = 0
    motorSpdFinal = 0

    for i in range(NMotor):
        tau = (i + 1) * TMotor
        # 使用 NLPSolver 中的 integrator（简单二阶积分）作为“植物”
        xInterpolate = integrator(x, u, tau)
        motorSpd = mod.motorVel(xInterpolate[0:3], xInterpolate[3:6])
        deltaMotorAng = TMotor * motorSpd
        deltaActuatorLen += deltaMotorAng / 2444.16

        if i == NMotor - 1:
            motorSpdFinal = motorSpd

    actuatorLenNew = actuatorLenPrev + deltaActuatorLen
    actuatorVelNew = motorSpdFinal / 2444.16
    qNew = mod.jointAngles(actuatorLenNew)
    qDotNew = mod.jointVel(qNew, actuatorVelNew)

    return csd.vertcat(qNew, qDotNew)


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

        # ------ 底座 yaw 控制滑条 ------
        self.yaw_var = tk.DoubleVar()
        self.yaw_var.set(0.0)  # 初始 0 度
        yaw_scale = tk.Scale(
            controls,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.yaw_var,
            label="Base yaw [deg]",
            length=200,
        )
        yaw_scale.pack(side=tk.LEFT, padx=10)

        # ========== MPC / 仿真状态 ==========
        self.mode = Mode.LIFT
        self.extF = 1000.0        # LIFT 模式的外力（用于可视化箭头长度）
        self.dutyCycle = DutyCycle.S2_30

        self.TSim = 5.0
        self.NSim = int(self.TSim / Ts)

        # 目标末端位姿：仍然是 2D [x, z, phi]
        self.poseDesired = csd.vertcat(3.2, C.yGround, -0.8)
        self.qDesired = mod.inverseKinematics(self.poseDesired)

        # 初始状态 x = [q, qdot]
        self.x0 = csd.vertcat(1.0, -2.2, -1.8, 0.0, 0.0, 0.0)

        self.running = False
        self.k = 0
        self.x = self.x0

        # 末端轨迹记录：每帧存一个 tip 的 3D 坐标
        self.tip_traj = []

        # 初始画一帧
        q0 = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q0, t=0.0)

    # ---------- 运动学：计算关节点 3D 坐标 ----------
    def link_positions_3d(self, q):
        """
        用原来 2D 模型 [boom, arm, bucket] 在 x–z 平面上的几何，
        然后通过绕 Z 轴的 yaw 旋转放到 3D 里显示。
        """
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])

        # 先在 X–Z 平面（Y = 0）下计算 3 个连杆的几何
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

        # 底座 yaw：绕世界 Z 轴旋转
        try:
            yaw_deg = float(self.yaw_var.get())
        except Exception:
            yaw_deg = 0.0
        yaw = math.radians(yaw_deg)

        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ])

        pts_rot = pts @ R.T
        return pts_rot

    # ---------- 画一帧 ----------
    def draw_frame(self, q, t):
        pts = self.link_positions_3d(q)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

        # 记录末端轨迹
        tip = pts[-1].copy()
        self.tip_traj.append(tip)

        self.ax.cla()

        # 挖机连杆
        self.ax.plot(X, Y, Z, "-o", label="excavator arm")

        # 末端轨迹（虚线）
        if len(self.tip_traj) > 1:
            traj = np.vstack(self.tip_traj)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                         "--", linewidth=1, label="tip trajectory")

        # 外力方向箭头（在 tip 上）
        force_dir = None
        if self.mode == Mode.LIFT:
            # LIFT: 模拟重力/吊装，向下
            force_dir = np.array([0.0, 0.0, -1.0])
        elif self.mode == Mode.DIG and len(pts) >= 2:
            # DIG: 模拟挖掘阻力，沿最后一段连杆反方向
            last_seg = pts[-1] - pts[-2]
            norm = np.linalg.norm(last_seg)
            if norm > 1e-6:
                force_dir = -last_seg / norm

        if force_dir is not None and abs(self.extF) > 1e-6:
            # 粗略长度缩放：extF=1000 → 约 2 m
            arrow_len = 0.002 * abs(self.extF)
            self.ax.quiver(
                tip[0], tip[1], tip[2],
                force_dir[0], force_dir[1], force_dir[2],
                length=arrow_len,
                normalize=True,
                label="external force",
            )

        L = TOTAL_LEN + 0.5
        # 画一条“地面线”（z = yGround）
        self.ax.plot(
            [-L, L],
            [0, 0],
            [float(C.yGround), float(C.yGround)],
            "k--",
            linewidth=1,
            label="ground",
        )

        # 对称范围，方便看底座旋转后的姿态
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
        """
        做一步 MPC：
        1) 用当前 x 建 NLP
        2) solveNLP 得到 u0
        3) 用 motor_commands 推进到 x_{k+1}

        这里保留错误信息，让你在终端里能看到具体异常。
        """
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
            u0 = sol.value(opti.u[:, 0])  # (3,)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Extract u0 failed at k={self.k}: {e}")
            return False

        u0_dm = csd.DM(u0)

        # 用 motor_commands 推进一整个 Ts
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
            # 保留前面设置的具体错误信息，不再覆盖
            self.running = False
            self.start_button.config(text="▶ Start (Realtime MPC)")
            return

        t = self.k * Ts
        q = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q, t)

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
        self.x = self.x0
        # 重置轨迹
        self.tip_traj = []
        q0 = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q0, t=0.0)
        self.status_var.set("Reset to initial state.")


def main():
    root = tk.Tk()
    app = ExcavatorRealtimeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
