import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D)
import math
import excavatorConstants as C

# Link lengths taken from excavatorConstants (based on old URDF)
LEN_BA = float(C.lenBA)
LEN_AL = float(C.lenAL)
LEN_LM = float(C.lenLM)
TOTAL_LEN = LEN_BA + LEN_AL + LEN_LM

# Try to use OpenCV if available (for video creation)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("visualisation.py: OpenCV (cv2) not installed – video export disabled.")

# === Output folder configuration ===
# Absolute path to this script's directory
SCRIPT_DIR = os.path.dirname(__file__)

# Store all frames/plots inside a "Plots" folder next to this file
visFolder = os.path.join(SCRIPT_DIR, "Plots")
os.makedirs(visFolder, exist_ok=True)
print(f"visualisation.py: Output folder = {visFolder}")


def _to_numpy(x):
    """Convert CasADi / list / array to 1D NumPy array of floats."""
    if x is None:
        return None
    arr = np.array(x, dtype=float)
    return arr


def visualise(x, _, q_desired, t, k, ext_force):
    """
    Save a 3D snapshot of the excavator configuration for frame k.

    This uses a simple 3-link planar model (boom–arm–bucket) with link
    lengths taken from excavatorConstants (LEN_BA, LEN_AL, LEN_LM).
    The arm lies in the X–Z plane (Y = 0) and is drawn as a 3D skeleton.

    Parameters
    ----------
    x : state vector [q1, q2, q3, q1dot, q2dot, q3dot]
        May be CasADi DM or NumPy.
    _ : unused (kept for compatibility with original signature)
    q_desired : desired joint configuration (optional)
    t : float, current time [s]
    k : int, frame index
    ext_force : external force (ignored here, kept for compatibility)
    """
    def _link_positions_3d(q):
        """Return 4×3 array of [x, y, z] for base, boom end, arm end, tip."""
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])

        # base
        p0 = np.array([0.0, 0.0, 0.0])

        # boom end
        p1 = p0 + np.array([LEN_BA * math.cos(alpha), 0.0,
                            LEN_BA * math.sin(alpha)])

        # arm end
        p2 = p1 + np.array([LEN_AL * math.cos(alpha + beta), 0.0,
                            LEN_AL * math.sin(alpha + beta)])

        # bucket tip
        p3 = p2 + np.array([LEN_LM * math.cos(alpha + beta + gamma), 0.0,
                            LEN_LM * math.sin(alpha + beta + gamma)])

        return np.vstack([p0, p1, p2, p3])

    # Convert state to NumPy
    x_np = _to_numpy(x).reshape(-1)
    q = x_np[:3]

    pts = _link_positions_3d(q)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"Excavator (3D skeleton)  t = {t:.2f} s  frame = {k}")

    # Plot current configuration
    ax.plot(X, Y, Z, "-o", label="current")

    # Optionally plot desired configuration as dashed line
    if q_desired is not None:
        try:
            qd = _to_numpy(q_desired).reshape(-1)[:3]
            pts_d = _link_positions_3d(qd)
            Xd, Yd, Zd = pts_d[:, 0], pts_d[:, 1], pts_d[:, 2]
            ax.plot(Xd, Yd, Zd, "--o", label="desired")
        except Exception:
            pass

    # Draw a simple ground line along X at Z=0
    L = TOTAL_LEN + 0.5
    ax.plot([0, L], [0, 0], [0, 0], "k--", linewidth=1)

    # Axis limits so the whole arm stays in view
    ax.set_xlim(0, L)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.5, L)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper left")

    save_path = os.path.join(visFolder, f"Excavator_{k}.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    if k % 10 == 0:
        print(f"visualisation.py: Saved 3D frame {k} to {save_path}")


def _pick_series(x=None, u=None, motorTorque=None, motorVel=None, motorPower=None):
    """
    Helper: choose one of the provided arrays as the main time series.
    """
    if x is not None:
        return np.array(x, dtype=float), "x"
    if u is not None:
        return np.array(u, dtype=float), "u"
    if motorTorque is not None:
        return np.array(motorTorque, dtype=float), "motorTorque"
    if motorVel is not None:
        return np.array(motorVel, dtype=float), "motorVel"
    if motorPower is not None:
        return np.array(motorPower, dtype=float), "motorPower"
    return None, None


def graph(t_start, t_end, interval, title, ylabel,
          x=None, u=None, motorTorque=None, motorVel=None, motorPower=None):
    """
    Generic time-series plotting function used by MPCSimulation.

    It expects one of x, u, motorTorque, motorVel, motorPower to be non-None.
    Each is interpreted as a (n_series, N_steps) or (N_steps, n_series) array.
    """
    series, key = _pick_series(
        x=x,
        u=u,
        motorTorque=motorTorque,
        motorVel=motorVel,
        motorPower=motorPower,
    )

    if series is None:
        print(f"[graph] No data provided for '{title}', skipping.")
        return

    series = np.array(series, dtype=float)
    # Make sure shape is (n_series, N_steps)
    if series.ndim == 1:
        series = series.reshape(1, -1)
    elif series.shape[0] > series.shape[1]:
        # Assume we want rows = series
        pass
    else:
        # Heuristic: if we have fewer rows than columns, treat rows as series
        pass

    n_steps = series.shape[1]
    t = np.linspace(t_start, t_end, n_steps)

    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(series.shape[0]):
        ax.plot(t, series[i, :], linewidth=1, label=f"{key}[{i}]")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Default legend outside if many series
    if series.shape[0] <= 3:
        ax.legend()
    else:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    save_path = os.path.join(visFolder, f"Graph_{title.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"visualisation.py: Saved graph '{title}' to {save_path}")


def plotMotorOpPt(motorTorque, motorVel, dutyCycle):
    """
    Plot motor operating points (torque vs velocity) and save to Plots/.
    """
    from excavatorModel import motorTorqueLimit

    motorTorque = np.array(motorTorque, dtype=float)
    motorVel = np.array(motorVel, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Draw torque limit curve
    angVel = np.linspace(0, 471.24, 200)
    torqueLimit = [motorTorqueLimit(w, dutyCycle) for w in angVel]
    ax.plot(angVel, torqueLimit, "r--", linewidth=1, label="Torque limit")

    # Plot operating points for each joint
    n_joints = motorTorque.shape[0]
    labels = ["Boom", "Arm", "Bucket"]
    for j in range(n_joints):
        ax.plot(
            np.abs(motorVel[j, :]),
            np.abs(motorTorque[j, :]),
            linewidth=1,
            label=labels[j] if j < len(labels) else f"Joint {j}",
        )

    ax.set_title(f"Motor operating points ({dutyCycle})")
    plt.xlabel(r"Motor velocity ($\mathsf{rad\ s^{-1}}$)")
    plt.ylabel("Motor torque (Nm)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    save_path = os.path.join(visFolder, "Motor_Operating_Points.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"visualisation.py: Saved motor operating plot to {save_path}")


def createVideo(k_start, k_end, name, fps):
    """
    Create an MP4 video from saved Excavator_k.jpg frames in visFolder.
    """
    if not HAS_CV2:
        print("createVideo: cv2 not available, skipping video creation.")
        return

    # Collect existing frames
    frame_files = []
    for k in range(k_start, k_end + 1):
        frame_path = os.path.join(visFolder, f"Excavator_{k}.jpg")
        if os.path.exists(frame_path):
            frame_files.append(frame_path)

    if not frame_files:
        print("createVideo: No frames found, skipping.")
        return

    # Read first frame to get size
    first = cv2.imread(frame_files[0])
    if first is None:
        print("createVideo: Failed to read first frame, aborting.")
        return
    height, width, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(visFolder, f"{name}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for k in range(k_start, k_end):
        frame_path = os.path.join(visFolder, f"Excavator_{k}.jpg")
        if not os.path.exists(frame_path):
            continue
        img = cv2.imread(frame_path)
        if img is None:
            continue
        # Ensure size consistent
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    print(f"visualisation.py: Video saved to: {video_path}")
