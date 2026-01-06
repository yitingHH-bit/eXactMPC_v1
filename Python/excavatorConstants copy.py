import numpy as np

# =========================
# 原始挖掘机 2D 平面几何常量
# （从 MATLAB 版 excavatorConstants 直接翻译）
# =========================

# --- Boom 几何 ---
iBC = np.array([0.135, -0.264])      # 机身上缸座 C 相对基座坐标
bBA = np.array([2.050, 0.0])         # 关节 A 相对基座坐标
bBD = np.array([1.025, 0.278])       # 点 D（缸铰点）相对基座
bBE = np.array([0.977, 0.737])       # 点 E（连杆铰点）相对基座
bDA = bBA - bBD
bAE = bBE - bBA
bCoMBoom = np.array([1.025, 0.384])  # Boom 质心相对基座

lenBC = float(np.linalg.norm(iBC))
lenBA = float(np.linalg.norm(bBA))
lenBD = float(np.linalg.norm(bBD))
lenBE = float(np.linalg.norm(bBE))
lenDA = float(np.linalg.norm(bDA))
lenAE = float(np.linalg.norm(bAE))
lenBCoMBoom = float(np.linalg.norm(bCoMBoom))

# 角度（弧度）
angABD = float(
    np.arccos((lenBA**2 + lenBD**2 - lenDA**2) / (2.0 * lenBA * lenBD))
)
angABCoMBoom = float(np.arctan2(bCoMBoom[1], bCoMBoom[0]))

# 质量与转动惯量（关于关节轴）
massBoom = 227.343
moiBoom = 67.768

# --- Arm 几何 ---
aAF = np.array([-0.251, 0.158])
aAG = np.array([-0.134, 0.320])
aAJ = np.array([0.880, 0.0])
aAL = np.array([1.050, 0.0])
aFL = aAL - aAF
aJG = aAG - aAJ
aJL = aAL - aAJ
aCoMArm = np.array([0.225, 0.227])

lenHJ = 0.240
lenHK = 0.240
lenAF = float(np.linalg.norm(aAF))
lenAG = float(np.linalg.norm(aAG))
lenAJ = float(np.linalg.norm(aAJ))
lenAL = float(np.linalg.norm(aAL))
lenFL = float(np.linalg.norm(aFL))
lenJG = float(np.linalg.norm(aJG))
lenJL = float(np.linalg.norm(aJL))
lenACoMArm = float(np.linalg.norm(aCoMArm))

angFAL = float(
    np.arccos((lenAL**2 + lenAF**2 - lenFL**2) / (2.0 * lenAF * lenAL))
)
angLACoMArm = float(np.arctan2(aCoMArm[1], aCoMArm[0]))

massArm = 130.123
moiArm = 30.258

# --- Bucket 几何 ---
lLK = np.array([-0.014, 0.164])
lLM = np.array([0.567, 0.0])
lKM = lLM - lLK
lCoMBucket = np.array([0.289, 0.166])

lenLK = float(np.linalg.norm(lLK))
lenLM = float(np.linalg.norm(lLM))
lenKM = float(np.linalg.norm(lKM))
lenLCoMBucket = float(np.linalg.norm(lCoMBucket))

angKLM = float(
    np.arccos((lenLM**2 + lenLK**2 - lenKM**2) / (2.0 * lenLK * lenLM))
)
angMLCoMBucket = float(np.arctan2(lCoMBucket[1], lCoMBucket[0]))

massBucket = 53.000
moiBucket = 3.021

# --- 环境 ---
yGround = -0.95737
g_vec = np.array([0.0, -9.81, 0.0])

# 兼容旧代码：有的地方直接用 C.g
g = g_vec

# =========================
# 新增：简化 3-DOF 链模型参数
# （给 2D/3D 动力学用）
# =========================

# 三段杆长（关节间距离）：
# 直接复用老模型的几何：
#   关节 0→1 : lenBA
#   关节 1→2 : lenAL
#   关节 2→3 : lenLM
L1 = lenBA     # 第一段（Boom）
L2 = lenAL     # 第二段（Arm）
L3 = lenLM     # 第三段（Bucket / Tool）

# 各段质心到本段关节的距离（沿杆方向）
# 这里用之前算好的质心模长做近似
lc1 = lenBCoMBoom       # Boom 质心
lc2 = lenACoMArm        # Arm 质心
lc3 = lenLCoMBucket     # Bucket 质心

# 质量（重复使用原来的 massBoom / massArm / massBucket）
m1 = massBoom
m2 = massArm
m3 = massBucket

# 各段绕出平面轴的转动惯量（近似用原来的 moiBoom 等）
I1 = moiBoom
I2 = moiArm
I3 = moiBucket

# =========================
# 新增：可旋转底座的转动惯量（yaw）
# 对应 excavatorModel.py 里的 C.Iz_base
# =========================
# 这里给一个合理的近似值，你以后可以用 CAD 或 URDF 精算后替换。
# 下车 + 上车两块大件的 izz 大概在 0.12 左右（来自 URDF 的示意），
# 再稍微加一点附件的惯量，取：
Iz_base = 0.13  # [kg·m^2] 仅作示意，可按需要修改

# 如果你后面要做完全 3D 刚体动力学，通常还会再定义：
#  - baseMass
#  - 其他方向惯量 Ix_base, Iy_base 等
# 到时候可以在这里继续扩展。
baseMass = 25.0  # 例如: lower_carriage(10kg) + upper_carriage(15kg) 的粗略和

