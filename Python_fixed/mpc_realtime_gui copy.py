import math
import tkinter as tk
from tkinter import ttk

import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import casadi as csd

import excavatorConstants as C
import excavatorModel as mod
from excavatorModel import DutyCycle
from NLPSolver import NLP, Mode, Ts, integrator


# -----------------------------
# Small numeric helpers
# -----------------------------
def _dm_to_np(x):
    """Convert CasADi DM/MX or list/np to 1D numpy float."""
    try:
        if isinstance(x, csd.DM):
            return np.array(x.full(), dtype=float).reshape(-1)
        if "casadi" in str(type(x)).lower():
            return np.array(csd.DM(x).full(), dtype=float).reshape(-1)
    except Exception:
        pass
    return np.asarray(x, dtype=float).reshape(-1)


def _is_finite(x) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(np.isfinite(x)))


def _clip(x, lo, hi):
    return float(np.clip(float(x), float(lo), float(hi)))


def _apply_T_np(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """T: (4,4), P: (N,3) -> (N,3)"""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if P.size == 0:
        return P.reshape(0, 3)
    T = np.asarray(T, dtype=float).reshape(4, 4)
    P_h = np.c_[P, np.ones((P.shape[0], 1), dtype=float)]
    Q = (T @ P_h.T).T
    return Q[:, :3]


def _nice_ticks(L):
    """Return nice ticks for [-L, L]."""
    L = float(L)
    # choose ~6-8 intervals
    step = max(0.2, round(L / 4.0, 1))
    # make step nicer
    if step > 1.0:
        step = round(step * 2) / 2.0
    ticks = np.arange(-L, L + 1e-9, step)
    return ticks


# Joint limits from URDF (model2_simple)
YAW_MIN, YAW_MAX = -8.0285146, 8.0285146
Q1_MIN, Q1_MAX = 0.0, 1.3962634
Q2_MIN, Q2_MAX = -0.5235988, 1.3962634
Q3_MIN, Q3_MAX = -2.0943951, 0.0


class ExcavatorRealtimeGUI:
    """
    Simplified UI version:
      - No sliders / manual mode
      - Multi-row input panel
      - URDF-consistent visualization kinematics (for "can extend flat" look)
      - Plant step matches NLP integrator (avoid mismatch)
      - Show desired target (star) and MPC predicted terminal pose (X)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Excavator MPC – Realtime 3D Simulation")

        # ----- derived reach -----
        self.L1 = float(C.L1)
        self.L2 = float(C.L2)
        self.tcp_len = float(math.hypot(float(C.TCP_DX), float(C.TCP_DZ)))
        self.reach_max = self.L1 + self.L2 + self.tcp_len

        # ----- 3D canvas -----
        self.fig = Figure(figsize=(7.2, 5.4))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # =========================
        # Controls (3 rows)
        # =========================
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        row1 = ttk.Frame(controls)
        row1.pack(side=tk.TOP, fill=tk.X)

        row2 = ttk.Frame(controls)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        row3 = ttk.Frame(controls)
        row3.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        # -------- Row 1: run + status + view --------
        self.start_button = ttk.Button(row1, text="▶ Start (Realtime MPC)", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=4)

        self.stop_button = ttk.Button(row1, text="⏸ Stop", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=4)

        self.reset_button = ttk.Button(row1, text="⟲ Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(row1, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        self.show_gripper_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="🖐 Gripper", variable=self.show_gripper_var).pack(side=tk.LEFT, padx=8)

        # -------- Row 2: init + extF --------
        self.yaw_init_deg = tk.DoubleVar(value=0.0)
        ttk.Label(row2, text="Yaw init [deg]:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.yaw_init_deg, width=8).pack(side=tk.LEFT, padx=3)

        self.q1_init_var = tk.DoubleVar(value=0.6)
        self.q2_init_var = tk.DoubleVar(value=-0.3)
        self.q3_init_var = tk.DoubleVar(value=-0.6)

        ttk.Label(row2, text="q1 init [rad]:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(row2, textvariable=self.q1_init_var, width=8).pack(side=tk.LEFT, padx=3)

        ttk.Label(row2, text="q2 init [rad]:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(row2, textvariable=self.q2_init_var, width=8).pack(side=tk.LEFT, padx=3)

        ttk.Label(row2, text="q3 init [rad]:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(row2, textvariable=self.q3_init_var, width=8).pack(side=tk.LEFT, padx=3)

        self.extF = 1000.0
        self.extF_var = tk.DoubleVar(value=self.extF)
        ttk.Label(row2, text="ExtF [N]:").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Entry(row2, textvariable=self.extF_var, width=10).pack(side=tk.LEFT, padx=3)

        ttk.Button(row2, text="↺ Apply inputs", command=self.apply_inputs).pack(side=tk.LEFT, padx=12)

        # -------- Row 3: target --------
        self.target_x_var = tk.DoubleVar(value=0.70)
        self.target_z_var = tk.DoubleVar(value=float(C.yGround))
        self.target_phi_deg_var = tk.DoubleVar(value=0.0)

        ttk.Label(row3, text="Target X [m]:").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.target_x_var, width=10).pack(side=tk.LEFT, padx=3)

        ttk.Label(row3, text="Target Z [m]:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(row3, textvariable=self.target_z_var, width=10).pack(side=tk.LEFT, padx=3)

        ttk.Label(row3, text="phi [deg]:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(row3, textvariable=self.target_phi_deg_var, width=10).pack(side=tk.LEFT, padx=3)

        ttk.Button(row3, text="🎯 Set target", command=self.set_target_pose).pack(side=tk.LEFT, padx=12)

        # ============ simplified gripper ============
        self.gripper_gap = 0.06
        self.gripper_finger_len = 0.10

        # ============ MPC config ============
        self.mode = Mode.LIFT
        self.dutyCycle = DutyCycle.S2_30

        self.TSim = 6.0
        self.NSim = int(self.TSim / Ts)

        self.poseDesired = np.array([0.70, float(C.yGround), 0.0], dtype=float).reshape(3,)
        self.pred_pose_last = None
        self.pred_yaw_last = None

        self.running = False
        self.k = 0

        # Optional metrics logging (enabled with env MPC_LOG_RESULTS=1)
        self.log_metrics = bool(int(os.getenv("MPC_LOG_RESULTS", "0")))
        self._metrics_saved = False
        self._init_metrics_buffers()

        # state init
        self._reset_state_from_inputs()

        # solver
        self.opti = NLP(self.mode, self.extF, self.dutyCycle)

        # ============ Monitor panel ============
        info = ttk.LabelFrame(self.root, text="Monitor (Desired / Predicted / Current)")
        info.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.t_var = tk.StringVar(value="0.00")
        self.k_var = tk.StringVar(value="0")

        self.cur_x_var = tk.StringVar(value="—")
        self.cur_z_var = tk.StringVar(value="—")
        self.cur_phi_var = tk.StringVar(value="—")

        self.des_x_var = tk.StringVar(value="—")
        self.des_z_var = tk.StringVar(value="—")
        self.des_phi_var = tk.StringVar(value="—")

        self.pred_x_var = tk.StringVar(value="—")
        self.pred_z_var = tk.StringVar(value="—")
        self.pred_phi_var = tk.StringVar(value="—")

        self.err_x_var = tk.StringVar(value="—")
        self.err_z_var = tk.StringVar(value="—")
        self.err_phi_var = tk.StringVar(value="—")

        ttk.Label(info, text="t [s]:").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.t_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="k:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Label(info, textvariable=self.k_var, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(info, text="Desired (x,z,phi):").grid(row=1, column=0, sticky="w")
        ttk.Label(info, textvariable=self.des_x_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(info, textvariable=self.des_z_var, width=10).grid(row=1, column=2, sticky="w")
        ttk.Label(info, textvariable=self.des_phi_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(info, text="Pred end (x,z,phi):").grid(row=2, column=0, sticky="w")
        ttk.Label(info, textvariable=self.pred_x_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(info, textvariable=self.pred_z_var, width=10).grid(row=2, column=2, sticky="w")
        ttk.Label(info, textvariable=self.pred_phi_var, width=10).grid(row=2, column=3, sticky="w")

        ttk.Label(info, text="Current (x,z,phi):").grid(row=3, column=0, sticky="w")
        ttk.Label(info, textvariable=self.cur_x_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(info, textvariable=self.cur_z_var, width=10).grid(row=3, column=2, sticky="w")
        ttk.Label(info, textvariable=self.cur_phi_var, width=10).grid(row=3, column=3, sticky="w")

        ttk.Label(info, text="Error (x,z,phi):").grid(row=4, column=0, sticky="w")
        ttk.Label(info, textvariable=self.err_x_var, width=10).grid(row=4, column=1, sticky="w")
        ttk.Label(info, textvariable=self.err_z_var, width=10).grid(row=4, column=2, sticky="w")
        ttk.Label(info, textvariable=self.err_phi_var, width=10).grid(row=4, column=3, sticky="w")

        # trajectory cache
        self.tip_traj = []

        # init target + draw
        self.set_target_pose(silent=True)
        self.draw_frame(_dm_to_np(self.x), t=0.0)

    # -----------------------------
    # init/reset
    # -----------------------------
    def _init_metrics_buffers(self):
        self.hist_t = []
        self.hist_q = []
        self.hist_qdot = []
        self.hist_u = []
        self.hist_motor_vel = []
        self.hist_motor_tau = []

    def _reset_state_from_inputs(self):
        yaw0 = math.radians(float(self.yaw_init_deg.get()))
        yaw0 = _clip(yaw0, YAW_MIN, YAW_MAX)

        q1 = _clip(self.q1_init_var.get(), Q1_MIN, Q1_MAX)
        q2 = _clip(self.q2_init_var.get(), Q2_MIN, Q2_MAX)
        q3 = _clip(self.q3_init_var.get(), Q3_MIN, Q3_MAX)

        self.x = csd.DM([yaw0, q1, q2, q3, 0.0, 0.0, 0.0, 0.0]).reshape((8, 1))
        self._x_last_good = _dm_to_np(self.x)
        if self.log_metrics:
            self._init_metrics_buffers()
            self.hist_t.append(0.0)
            self.hist_q.append(np.array([q1, q2, q3], dtype=float))
            self.hist_qdot.append(np.array([0.0, 0.0, 0.0], dtype=float))
            self.hist_motor_vel.append(np.zeros(3, dtype=float))
            self.hist_motor_tau.append(np.zeros(3, dtype=float))

    # -----------------------------
    # Target handling
    # -----------------------------
    def set_target_pose(self, silent=False):
        try:
            x = float(self.target_x_var.get())
            z = float(self.target_z_var.get())
            phi_deg = float(self.target_phi_deg_var.get())
            phi = math.radians(phi_deg)
        except Exception as e:
            if not silent:
                self.status_var.set(f"Set target failed: {e}")
            return False

        self.poseDesired = np.array([x, z, phi], dtype=float).reshape(3,)

        self.des_x_var.set(f"{x:.3f}")
        self.des_z_var.set(f"{z:.3f}")
        self.des_phi_var.set(f"{phi_deg:.1f}")

        if not silent:
            self.status_var.set(f"Target set: x={x:.3f}, z={z:.3f}, phi={phi_deg:.1f} deg")
        return True

    def apply_inputs(self):
        # extF
        try:
            self.extF = float(self.extF_var.get())
        except Exception:
            self.status_var.set("ExtF parse failed.")
            return

        # reset state from yaw/q init
        self._reset_state_from_inputs()

        # rebuild solver (important: avoid "sticky" params)
        self.opti = NLP(self.mode, self.extF, self.dutyCycle)

        # update target from current entries
        self.set_target_pose(silent=True)

        # clear predicted marker & trail
        self.pred_pose_last = None
        self.pred_yaw_last = None
        self.pred_x_var.set("—")
        self.pred_z_var.set("—")
        self.pred_phi_var.set("—")
        self.tip_traj = []

        self.k = 0
        self.status_var.set(f"Applied inputs. ExtF={self.extF:.1f} N. Reset state + solver.")
        self.draw_frame(_dm_to_np(self.x), t=0.0)

    # -----------------------------
    # URDF-consistent visualization kinematics
    # -----------------------------
    def link_positions_3d(self, yaw, q):
        """
        Compute joint/world points: base -> joint1 -> joint2 -> TCP (p3), then yaw rotate around Z.

        IMPORTANT: URDF joints rotate around +Y.
        In X–Z plane, +Y rotation corresponds to rotating the vector by -angle.
        So we use:
            theta1 = OFF1 - q1
            theta2 = OFF2 - (q1+q2)
        And TCP offset is rotated by -(q1+q2+q3).
        """
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])
        phi = alpha + beta + gamma

        p0 = np.array([0.0, 0.0, 0.0], dtype=float)

        th1 = float(C.OFF1) - alpha
        p1 = p0 + np.array([self.L1 * math.cos(th1), 0.0, self.L1 * math.sin(th1)], dtype=float)

        th2 = float(C.OFF2) - (alpha + beta)
        p2 = p1 + np.array([self.L2 * math.cos(th2), 0.0, self.L2 * math.sin(th2)], dtype=float)

        # rotate TCP offset by -phi
        c = math.cos(phi)
        s = math.sin(phi)
        dx = float(C.TCP_DX) * c + float(C.TCP_DZ) * s
        dz = -float(C.TCP_DX) * s + float(C.TCP_DZ) * c
        p3 = p2 + np.array([dx, 0.0, dz], dtype=float)

        pts = np.vstack([p0, p1, p2, p3])

        yaw = float(yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)
        return pts @ Rz.T

    def _tool_T_world(self, yaw, q):
        """tool_body pose in world: origin at p2, rotation Rz(yaw) @ Ry(+phi)."""
        yaw = float(yaw)
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])
        phi = alpha + beta + gamma

        pts = self.link_positions_3d(yaw, q)
        p2 = pts[2].copy()

        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)

        c = math.cos(phi)
        s = math.sin(phi)
        Ry = np.array([[c, 0.0, s],
                       [0.0, 1.0, 0.0],
                       [-s, 0.0, c]], dtype=float)

        T = np.eye(4, dtype=float)
        T[:3, :3] = Rz @ Ry
        T[:3, 3] = p2
        return T

    # -----------------------------
    # Rendering
    # -----------------------------
    def draw_frame(self, x_vec, t):
        x_arr = np.asarray(x_vec, dtype=float).reshape(-1)
        yaw = float(x_arr[0])
        q = x_arr[1:4].astype(float)

        # current planar pose from model FK (consistent with NLP cost)
        try:
            pose_now = _dm_to_np(mod.forwardKinematics(csd.DM(q).reshape((3, 1))))
        except Exception:
            pose_now = np.array([], dtype=float)

        if pose_now.size == 3 and _is_finite(pose_now):
            self.cur_x_var.set(f"{pose_now[0]:.3f}")
            self.cur_z_var.set(f"{pose_now[1]:.3f}")
            self.cur_phi_var.set(f"{math.degrees(pose_now[2]):.1f}")

            des = self.poseDesired
            self.err_x_var.set(f"{(pose_now[0] - des[0]):+.3f}")
            self.err_z_var.set(f"{(pose_now[1] - des[1]):+.3f}")
            self.err_phi_var.set(f"{math.degrees(pose_now[2] - des[2]):+.1f}")
        else:
            self.cur_x_var.set("—")
            self.cur_z_var.set("—")
            self.cur_phi_var.set("—")
            self.err_x_var.set("—")
            self.err_z_var.set("—")
            self.err_phi_var.set("—")

        self.t_var.set(f"{t:.2f}")
        self.k_var.set(str(self.k))

        # 3D arm points
        pts = self.link_positions_3d(yaw, q)
        tip_w = pts[-1].copy()
        self.tip_traj.append(tip_w)

        self.ax.cla()
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", label="excavator arm")

        if len(self.tip_traj) > 1:
            traj = np.vstack(self.tip_traj)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "--", linewidth=1, label="tip trajectory")

        # target star (desired) in world using current yaw
        try:
            xd, zd = float(self.poseDesired[0]), float(self.poseDesired[1])
            xw = xd * math.cos(yaw)
            yw = xd * math.sin(yaw)
            self.ax.scatter([xw], [yw], [zd], marker="*", s=90, label="target")
        except Exception:
            pass

        # predicted terminal marker (from MPC)
        if self.pred_pose_last is not None and self.pred_yaw_last is not None:
            try:
                xd, zd = float(self.pred_pose_last[0]), float(self.pred_pose_last[1])
                yN = float(self.pred_yaw_last)
                xw = xd * math.cos(yN)
                yw = xd * math.sin(yN)
                self.ax.scatter([xw], [yw], [zd], marker="x", s=70, label="mpc_plan_end")
            except Exception:
                pass

        # gripper: two line segments in tool frame
        if self.show_gripper_var.get():
            try:
                T_tool = self._tool_T_world(yaw, q)
                gap = float(self.gripper_gap)
                finger_len = float(self.gripper_finger_len)
                g = 0.5 * gap

                bx = float(C.TCP_DX)
                bz = float(C.TCP_DZ)

                p0L = np.array([bx, +g, bz], dtype=float)
                p1L = p0L + np.array([finger_len, 0.0, 0.0], dtype=float)
                p0R = np.array([bx, -g, bz], dtype=float)
                p1R = p0R + np.array([finger_len, 0.0, 0.0], dtype=float)

                PwL = _apply_T_np(T_tool, np.vstack([p0L, p1L]))
                PwR = _apply_T_np(T_tool, np.vstack([p0R, p1R]))

                self.ax.plot(PwL[:, 0], PwL[:, 1], PwL[:, 2], linewidth=3, label="gripper")
                self.ax.plot(PwR[:, 0], PwR[:, 1], PwR[:, 2], linewidth=3, label="_nolegend_")
            except Exception:
                pass

        # ground line (z = yGround)
        L = max(self.reach_max + 0.3, 1.2)
        self.ax.plot([-L, L], [0, 0], [float(C.yGround), float(C.yGround)], "k--", linewidth=1, label="ground")

        # axis limits + ticks
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_zlim(float(C.yGround) - 0.3, L)

        ticks = _nice_ticks(L)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)

        zticks = _nice_ticks(L)
        # keep z ticks not too dense
        self.ax.set_zticks(zticks)

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title(f"Realtime MPC – t = {t:.2f} s  (k={self.k})")
        self.ax.view_init(elev=25, azim=-60)
        self.ax.legend(loc="upper left", fontsize=8)

        self.canvas.draw()

    # -----------------------------
    # MPC loop
    # -----------------------------
    def _record_metrics(self, t, x_now, u0):
        if not self.log_metrics:
            return
        q = np.asarray(x_now[1:4], dtype=float).reshape(3)
        qdot = np.asarray(x_now[5:8], dtype=float).reshape(3)
        u_j = np.asarray(u0[1:], dtype=float).reshape(3)
        try:
            motor_tau = _dm_to_np(mod.motorTorque(q, qdot, u_j, self.extF)).reshape(3)
        except Exception:
            motor_tau = np.zeros(3, dtype=float)
        try:
            motor_vel = _dm_to_np(mod.motorVel(q, qdot)).reshape(3)
        except Exception:
            motor_vel = np.zeros(3, dtype=float)

        self.hist_t.append(float(t))
        self.hist_q.append(q)
        self.hist_qdot.append(qdot)
        self.hist_u.append(u_j)
        self.hist_motor_vel.append(motor_vel)
        self.hist_motor_tau.append(motor_tau)

    def _save_metrics(self):
        if not self.log_metrics or self._metrics_saved:
            return
        try:
            np.savez(
                "mpc_results_gui.npz",
                time=np.asarray(self.hist_t, dtype=float),
                q=np.asarray(self.hist_q, dtype=float).T,
                qdot=np.asarray(self.hist_qdot, dtype=float).T,
                u=np.asarray(self.hist_u, dtype=float).T,
                motorVel=np.asarray(self.hist_motor_vel, dtype=float).T,
                motorTorque=np.asarray(self.hist_motor_tau, dtype=float).T,
                Ts=Ts,
            )
            self._metrics_saved = True
            self.status_var.set("Metrics saved to mpc_results_gui.npz")
        except Exception as e:
            self.status_var.set(f"Save metrics failed: {e}")

    def _plant_step(self, x_now_np: np.ndarray, u0_np: np.ndarray) -> np.ndarray:
        """Plant step that matches the NLP model integrator."""
        x_dm = csd.DM(x_now_np).reshape((8, 1))
        u_dm = csd.DM(u0_np).reshape((4, 1))
        x_next = integrator(x_dm, u_dm, Ts)
        x_next_np = _dm_to_np(x_next)

        if x_next_np.size >= 4:
            x_next_np[0] = _clip(x_next_np[0], YAW_MIN, YAW_MAX)
            x_next_np[1] = _clip(x_next_np[1], Q1_MIN, Q1_MAX)
            x_next_np[2] = _clip(x_next_np[2], Q2_MIN, Q2_MAX)
            x_next_np[3] = _clip(x_next_np[3], Q3_MIN, Q3_MAX)

        return x_next_np

    def step_mpc(self):
        if self.k >= self.NSim:
            self.status_var.set("Finished.")
            self._save_metrics()
            return False

        # validate target
        try:
            self.poseDesired = np.asarray(self.poseDesired, dtype=float).reshape(3,)
            if not _is_finite(self.poseDesired):
                raise ValueError("poseDesired not finite")
        except Exception:
            self.status_var.set("Invalid target. Please set target again.")
            return False

        # current state
        x_now = _dm_to_np(self.x)
        if x_now.size != 8 or (not _is_finite(x_now)):
            x_now = getattr(self, "_x_last_good", np.zeros(8, dtype=float))
        self._x_last_good = x_now.copy()

        # solve NLP
        try:
            sol = self.opti.solveNLP(x_now, self.poseDesired)
        except Exception as e:
            self.status_var.set(f"NLP solve failed: {e}")
            return False

        # extract u0
        try:
            u0 = sol.value(self.opti.u[:, 0])
            u0_np = np.asarray(u0, dtype=float).reshape(-1)
            if u0_np.size != 4 or (not _is_finite(u0_np)):
                raise ValueError("u0 invalid")
        except Exception as e:
            self.status_var.set(f"Extract u0 failed: {e}")
            return False

        # predicted terminal pose (debug)
        try:
            Xpred = sol.value(self.opti.x)
            Xpred = np.asarray(Xpred, dtype=float)
            qN = Xpred[1:4, -1].reshape(3, 1)
            yawN = float(Xpred[0, -1])
            poseN = _dm_to_np(mod.forwardKinematics(csd.DM(qN)))

            if poseN.size == 3 and _is_finite(poseN):
                self.pred_pose_last = poseN
                self.pred_yaw_last = yawN
                self.pred_x_var.set(f"{poseN[0]:.3f}")
                self.pred_z_var.set(f"{poseN[1]:.3f}")
                self.pred_phi_var.set(f"{math.degrees(poseN[2]):.1f}")
            else:
                self.pred_pose_last = None
                self.pred_yaw_last = None
                self.pred_x_var.set("—")
                self.pred_z_var.set("—")
                self.pred_phi_var.set("—")
        except Exception:
            self.pred_pose_last = None
            self.pred_yaw_last = None
            self.pred_x_var.set("—")
            self.pred_z_var.set("—")
            self.pred_phi_var.set("—")

        # plant step
        x_next_np = self._plant_step(x_now, u0_np)
        if x_next_np.size != 8 or (not _is_finite(x_next_np)):
            self.status_var.set("Plant step produced NaN/Inf.")
            return False

        self.x = csd.DM(x_next_np).reshape((8, 1))
        self._x_last_good = x_next_np.copy()

        t = self.k * Ts
        self._record_metrics(t, x_now, u0_np)

        self.k += 1
        return True

    def update_frame(self):
        if not self.running:
            return

        ok = self.step_mpc()
        if not ok:
            self.running = False
            self.start_button.config(text="▶ Start (Realtime MPC)")
            return

        t = self.k * Ts
        self.draw_frame(_dm_to_np(self.x), t)

        self.root.after(10, self.update_frame)

    # -----------------------------
    # UI callbacks
    # -----------------------------
    def start_simulation(self):
        if self.running:
            return
        if not self.set_target_pose(silent=False):
            return

        # rebuild solver every run (avoid old parameters sticking)
        self.opti = NLP(self.mode, self.extF, self.dutyCycle)

        self.running = True
        self.start_button.config(text="▶ Running…")
        self.status_var.set("Running realtime MPC…")
        self.update_frame()

    def stop_simulation(self):
        self.running = False
        self.start_button.config(text="▶ Start (Realtime MPC)")
        self.status_var.set("Paused.")

    def reset_simulation(self):
        self.stop_simulation()
        self.k = 0

        self.pred_pose_last = None
        self.pred_yaw_last = None
        self.pred_x_var.set("—")
        self.pred_z_var.set("—")
        self.pred_phi_var.set("—")

        self.tip_traj = []
        self._reset_state_from_inputs()

        self.draw_frame(_dm_to_np(self.x), t=0.0)
        self.status_var.set("Reset.")


def main():
    root = tk.Tk()
    app = ExcavatorRealtimeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
