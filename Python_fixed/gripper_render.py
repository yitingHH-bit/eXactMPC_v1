# gripper_render.py
# Lightweight OBJ point-cloud rendering of the end-effector gripper.
# No external dependencies besides numpy.

import os
import numpy as np


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx

def voxel_downsample(P, voxel=0.003):
    """voxel in meters; bigger -> fewer points"""
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return P.reshape(0, 3)
    key = np.floor(P / voxel).astype(np.int32)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]

def T_from_xyz_rpy(xyz, rpy):
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_R(np.array(rpy, dtype=float))
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def apply_T(T, P):
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return P.reshape(0, 3)
    P_h = np.c_[P, np.ones((P.shape[0], 1), dtype=float)]
    Q = (np.asarray(T, dtype=float) @ P_h.T).T
    return Q[:, :3]


def load_obj_vertices(obj_path, max_points=8000, seed=0):
    """Read only 'v x y z' vertices. Randomly downsample for speed."""
    verts = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except Exception:
                        continue
    V = np.asarray(verts, dtype=float)
    if V.shape[0] == 0:
        return V
    if V.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(V.shape[0], size=max_points, replace=False)
        V = V[idx]
    return V


class GripperMesh:
    """
    Build a point cloud of the gripper from OBJ meshes and URDF fixed joints.

    Frames (from your URDF):
        tool_body -> gripper_frame
        gripper_frame -> gripper_claw_1
        gripper_frame -> gripper_claw_2

    Use points_in_world(T_tool_world) to get world coordinates.
    """

    def __init__(self, mesh_dir="meshes", max_points_each=6000):
        self.mesh_dir = mesh_dir

        self.frame_obj = os.path.join(mesh_dir, "gripper_frame_Mesh.obj")
        self.c1_obj = os.path.join(mesh_dir, "gripper_claw_1_Mesh.obj")
        self.c2_obj = os.path.join(mesh_dir, "gripper_claw_2_Mesh.obj")

        self.V_frame = load_obj_vertices(self.frame_obj, max_points=max_points_each, seed=0)
        self.V_c1 = load_obj_vertices(self.c1_obj, max_points=max_points_each, seed=1)
        self.V_c2 = load_obj_vertices(self.c2_obj, max_points=max_points_each, seed=2)

        # ===== fixed joints from your URDF =====
        # tool_body -> gripper_frame
        self.T_gripper_tool = T_from_xyz_rpy(
            xyz=[0.0363731, 0.0, -0.0450618],
            rpy=[0.0, 0.0, 0.0],
        )
        # gripper_frame -> claw1
        self.T_c1_gripper = T_from_xyz_rpy(
            xyz=[0.0261671, 0.0, -0.0385303],
            rpy=[-0.3665192, 0.0, 1.5707964],
        )
        # gripper_frame -> claw2
        self.T_c2_gripper = T_from_xyz_rpy(
            xyz=[-0.0238304, 0.0, -0.0385303],
            rpy=[0.0, -0.3316125, -3.1415927],
        )

    def points_in_tool(self):
        pts = []
        if self.V_frame.size:
            pts.append(apply_T(self.T_gripper_tool, self.V_frame))
        if self.V_c1.size:
            T = self.T_gripper_tool @ self.T_c1_gripper
            pts.append(apply_T(T, self.V_c1))
        if self.V_c2.size:
            T = self.T_gripper_tool @ self.T_c2_gripper
            pts.append(apply_T(T, self.V_c2))
        if not pts:
            return np.zeros((0, 3), dtype=float)
        P = np.vstack(pts)
        P = voxel_downsample(P, voxel=0.003)   # 0.002~0.006 都可以试
        return P

    def points_in_world(self, T_tool_world):
        P_tool = self.points_in_tool()
        if P_tool.size == 0:
            return P_tool
        return apply_T(T_tool_world, P_tool)
