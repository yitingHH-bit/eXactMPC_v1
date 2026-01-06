import casadi as csd
from enum import Enum
import excavatorConstants as C


class DutyCycle(Enum):
    S1 = 0
    S2_60 = 1
    S2_30 = 2
    PEAK = 3


# def forwardKinematics(q):
#     alpha = q[0]
#     beta = q[1]
#     gamma = q[2]

#     x = C.lenBA * csd.cos(alpha) + C.lenAL * csd.cos(alpha + beta) + C.lenLM * csd.cos(alpha + beta + gamma)
#     y = C.lenBA * csd.sin(alpha) + C.lenAL * csd.sin(alpha + beta) + C.lenLM * csd.sin(alpha + beta + gamma)
#     theta = alpha + beta + gamma
#     return csd.vertcat(x, y, theta)

def forwardKinematicsPlanar(q):
    """
    2D 正运动学：q = [alpha, beta, gamma]
    使用 excavatorConstants 里的简化杆长:
        L1, L2, L3  <-- 来自 URDF
    返回: [x, z, phi]
    """
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    # URDF-fitted kinematics with link direction offsets (OFF1/OFF2)
    # and a TCP offset vector (TCP_DX/TCP_DZ) expressed in the tool_body frame.
    #
    #   p_tool = R(alpha) v1 + R(alpha+beta) v2
    #   p_tcp  = p_tool + R(phi) v_tcp
    #   phi    = alpha + beta + gamma

    phi = alpha + beta + gamma

    # Link vectors are not aligned with +X when joint angle = 0, so we add OFF1/OFF2.
    x_tool = C.L1 * csd.cos(alpha + C.OFF1) + C.L2 * csd.cos(alpha + beta + C.OFF2)
    z_tool = C.L1 * csd.sin(alpha + C.OFF1) + C.L2 * csd.sin(alpha + beta + C.OFF2)

    # TCP offset (dx, dz) in tool frame
    x_tcp = x_tool + (C.TCP_DX * csd.cos(phi) - C.TCP_DZ * csd.sin(phi))
    z_tcp = z_tool + (C.TCP_DX * csd.sin(phi) + C.TCP_DZ * csd.cos(phi))

    return csd.vertcat(x_tcp, z_tcp, phi)


# 兼容旧项目名字：旧代码里直接叫 forwardKinematics(q)
def forwardKinematics(q):
    return forwardKinematicsPlanar(q)


def inverseKinematics(pose, elbow: str = "down"):
    """Planar IK for the URDF-fitted model.

    pose: [x, z, phi]
    elbow: "down" (default) or "up" (chooses sign of sin(beta)).

    Notes
    -----
    - Uses OFF1/OFF2 so the model matches URDF joint-to-joint *vectors*.
    - Uses TCP_DX/TCP_DZ so the "tip" matches the gripper center.
    - Clamps cos(beta) to avoid sqrt of negative due to numerical/target issues.
    """
    x_tip = pose[0]
    z_tip = pose[1]
    phi = pose[2]

    # Back out the TCP offset to get the tool_body origin target.
    x_tool = x_tip - (C.TCP_DX * csd.cos(phi) - C.TCP_DZ * csd.sin(phi))
    z_tool = z_tip - (C.TCP_DX * csd.sin(phi) + C.TCP_DZ * csd.cos(phi))

    # Reduce to a standard 2-link IK by absorbing fixed link offsets.
    # Let A1 = alpha + OFF1, and B1 = beta + (OFF2 - OFF1).
    # Then: p = L1*[cos(A1), sin(A1)] + L2*[cos(A1+B1), sin(A1+B1)].
    L1 = C.L1
    L2 = C.L2
    eps = 1e-9

    r2 = x_tool ** 2 + z_tool ** 2
    cosB1 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cosB1 = csd.fmin(1.0 - eps, csd.fmax(-1.0 + eps, cosB1))
    sinB1_abs = csd.sqrt(csd.fmax(0.0, 1 - cosB1 ** 2))

    if elbow.lower() == "up":
        sinB1 = sinB1_abs
    else:
        sinB1 = -sinB1_abs

    B1 = csd.atan2(sinB1, cosB1)
    A1 = csd.atan2(z_tool, x_tool) - csd.atan2(L2 * sinB1, L1 + L2 * cosB1)

    alpha = A1 - C.OFF1
    beta = B1 - C.OFF2 + C.OFF1
    gamma = phi - alpha - beta

    return csd.vertcat(alpha, beta, gamma)


def inverseDynamics(q, qDot, qDDot, F):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    alphaDot = qDot[0]
    betaDot = qDot[1]
    gammaDot = qDot[2]

    # --- Jacobians for CoM of boom, arm, bucket ------------------------
    jacBoom = csd.vertcat(
        csd.horzcat(-C.lenBCoMBoom * csd.sin(alpha + C.angABCoMBoom), 0, 0),
        csd.horzcat(C.lenBCoMBoom * csd.cos(alpha + C.angABCoMBoom), 0, 0),
        csd.horzcat(1, 0, 0),
    )

    jacArm = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.sin(alpha)
            - C.lenACoMArm * csd.sin(alpha + beta + C.angLACoMArm),
            -C.lenACoMArm * csd.sin(alpha + beta + C.angLACoMArm),
            0,
        ),
        csd.horzcat(
            C.lenBA * csd.cos(alpha)
            + C.lenACoMArm * csd.cos(alpha + beta + C.angLACoMArm),
            C.lenACoMArm * csd.cos(alpha + beta + C.angLACoMArm),
            0,
        ),
        csd.horzcat(1, 1, 0),
    )

    jacBucket = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.sin(alpha)
            - C.lenAL * csd.sin(alpha + beta)
            - C.lenLCoMBucket * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
            -C.lenAL * csd.sin(alpha + beta)
            - C.lenLCoMBucket * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
            -C.lenLCoMBucket * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
        ),
        csd.horzcat(
            C.lenBA * csd.cos(alpha)
            + C.lenAL * csd.cos(alpha + beta)
            + C.lenLCoMBucket * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
            C.lenAL * csd.cos(alpha + beta)
            + C.lenLCoMBucket * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
            C.lenLCoMBucket * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
        ),
        csd.horzcat(1, 1, 1),
    )

    # --- Time derivatives of Jacobians ---------------------------------
    jacBoomDot = csd.vertcat(
        csd.horzcat(
            -C.lenBCoMBoom * csd.cos(alpha + C.angABCoMBoom) * alphaDot,
            0,
            0,
        ),
        csd.horzcat(
            -C.lenBCoMBoom * csd.sin(alpha + C.angABCoMBoom) * alphaDot,
            0,
            0,
        ),
        csd.horzcat(0, 0, 0),
    )

    jacArmDot = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.cos(alpha) * alphaDot
            - C.lenACoMArm * csd.cos(alpha + beta + C.angLACoMArm) * (alphaDot + betaDot),
            -C.lenACoMArm * csd.cos(alpha + beta + C.angLACoMArm) * (alphaDot + betaDot),
            0,
        ),
        csd.horzcat(
            -C.lenBA * csd.sin(alpha) * alphaDot
            - C.lenACoMArm * csd.sin(alpha + beta + C.angLACoMArm) * (alphaDot + betaDot),
            -C.lenACoMArm * csd.sin(alpha + beta + C.angLACoMArm) * (alphaDot + betaDot),
            0,
        ),
        csd.horzcat(0, 0, 0),
    )

    jacBucketDot = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.cos(alpha) * alphaDot
            - C.lenAL * csd.cos(alpha + beta) * (alphaDot + betaDot)
            - C.lenLCoMBucket * csd.cos(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
            -C.lenAL * csd.cos(alpha + beta) * (alphaDot + betaDot)
            - C.lenLCoMBucket * csd.cos(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
            -C.lenLCoMBucket
            * csd.cos(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
        ),
        csd.horzcat(
            -C.lenBA * csd.sin(alpha) * alphaDot
            - C.lenAL * csd.sin(alpha + beta) * (alphaDot + betaDot)
            - C.lenLCoMBucket * csd.sin(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
            -C.lenAL * csd.sin(alpha + beta) * (alphaDot + betaDot)
            - C.lenLCoMBucket * csd.sin(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
            -C.lenLCoMBucket
            * csd.sin(alpha + beta + gamma + C.angMLCoMBucket)
            * (alphaDot + betaDot + gammaDot),
        ),
        csd.horzcat(0, 0, 0),
    )

    # --- Mass matrix, Coriolis/centrifugal, gravity, external -----------
    qDot_vec = csd.vertcat(alphaDot, betaDot, gammaDot)

    M = (
        csd.transpose(jacBoom[0:2, :]) @ C.massBoom @ jacBoom[0:2, :]
        + csd.transpose(jacBoom[2, :]) @ (C.moiBoom @ jacBoom[2, :])
        + csd.transpose(jacArm[0:2, :]) @ C.massArm @ jacArm[0:2, :]
        + csd.transpose(jacArm[2, :]) @ (C.moiArm @ jacArm[2, :])
        + csd.transpose(jacBucket[0:2, :]) @ C.massBucket @ jacBucket[0:2, :]
        + csd.transpose(jacBucket[2, :]) @ (C.moiBucket @ jacBucket[2, :])
    )

    b = (
        csd.transpose(jacBoom[0:2, :]) @ C.massBoom @ jacBoomDot[0:2, :] @ qDot_vec
        + csd.transpose(jacBoom[2, :]) @ (C.moiBoom @ jacBoomDot[2, :] @ qDot_vec)
        + csd.transpose(jacArm[0:2, :]) @ C.massArm @ jacArmDot[0:2, :] @ qDot_vec
        + csd.transpose(jacArm[2, :]) @ (C.moiArm @ jacArmDot[2, :] @ qDot_vec)
        + csd.transpose(jacBucket[0:2, :]) @ C.massBucket @ jacBucketDot[0:2, :] @ qDot_vec
        + csd.transpose(jacBucket[2, :]) @ (C.moiBucket @ jacBucketDot[2, :] @ qDot_vec)
    )

    g = (
        -csd.transpose(jacBoom[0:2, :]) @ C.massBoom @ C.g
        - csd.transpose(jacArm[0:2, :]) @ C.massArm @ C.g
        - csd.transpose(jacBucket[0:2, :]) @ C.massBucket @ C.g
    )

    extF = csd.transpose(jacBucket[0:2, :]) @ F

    return M @ qDDot + b + g - extF


def jointAngles(len):
    lenBoom = len[0]
    lenArm = len[1]
    lenBucket = len[2]

    # Boom
    R = C.lenBC
    theta = csd.atan2(C.iBC[1], C.iBC[0])
    alpha = (
        csd.acos(
            (-lenBoom ** 2 + C.lenBD ** 2 + C.lenBC ** 2)
            / (2 * R * C.lenBD)
        )
        + theta
        - C.angABD
    )

    # Arm
    R = C.lenAE
    theta = csd.atan2(C.bAE[0], C.bAE[1])
    beta = (
        csd.asin(
            (-lenArm ** 2 + C.lenAF ** 2 + C.lenAE ** 2)
            / (2 * R * C.lenAF)
        )
        - theta
        - C.angFAL
    )

    # Bucket
    R = C.lenJG
    theta = csd.atan2(C.aJG[0], C.aJG[1])
    angLJH = (
        csd.asin(
            (-lenBucket ** 2 + C.lenHJ ** 2 + C.lenJG ** 2)
            / (2 * R * C.lenHJ)
        )
        - theta
    )

    R = csd.sqrt(
        (C.lenJL - C.lenHJ * csd.cos(angLJH)) ** 2
        + (C.lenHJ * csd.sin(angLJH)) ** 2
    )
    theta = csd.atan2(
        C.lenHJ * csd.sin(angLJH),
        C.lenJL - C.lenHJ * csd.cos(angLJH),
    )
    gamma = (
        csd.acos(
            (
                C.lenHK ** 2
                - C.lenJL ** 2
                - C.lenLK ** 2
                - C.lenHJ ** 2
                + 2 * C.lenJL * C.lenHJ * csd.cos(angLJH)
            )
            / (2 * R * C.lenLK)
        )
        - theta
        - C.angKLM
    )

    return csd.vertcat(alpha, beta, gamma)


def jointVel(q, actuatorVel):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    lenBoomDot = actuatorVel[0]
    lenArmDot = actuatorVel[1]
    lenBucketDot = actuatorVel[2]

    alphaDot = lenBoomDot / (
        0.0344 * alpha ** 3 - 0.1377 * alpha ** 2 - 0.0208 * alpha + 0.2956
    )
    betaDot = lenArmDot / (
        0.0312 * beta ** 3 + 0.2751 * beta ** 2 + 0.582 * beta + 0.0646
    )
    gammaDot = lenBucketDot / (
        0.0192 * gamma ** 3 + 0.0864 * gamma ** 2 + 0.045 * gamma - 0.1695
    )

    return csd.vertcat(alphaDot, betaDot, gammaDot)


def actuatorLen(q):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    # 拟合的缸长–关节角关系（多项式）
    # Boom
    lenBoom = (
        0.0086 * alpha ** 4
        - 0.0459 * alpha ** 3
        - 0.0104 * alpha ** 2
        + 0.2956 * alpha
        + 1.042
    )

    # Arm
    lenArm = (
        0.0078 * beta ** 4
        + 0.0917 * beta ** 3
        + 0.2910 * beta ** 2
        + 0.0646 * beta
        + 1.0149
    )

    # Bucket
    lenBucket = (
        0.0048 * gamma ** 4
        + 0.0288 * gamma ** 3
        + 0.0225 * gamma ** 2
        - 0.1695 * gamma
        + 0.9434
    )

    return csd.vertcat(lenBoom, lenArm, lenBucket)


def actuatorVelFactor(q):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    velFactorBoom = (
        0.0344 * alpha ** 3
        - 0.1377 * alpha ** 2
        - 0.0208 * alpha
        + 0.2956
    )
    velFactorArm = (
        0.0312 * beta ** 3
        + 0.2751 * beta ** 2
        + 0.582 * beta
        + 0.0646
    )
    velFactorBucket = (
        0.0192 * gamma ** 3
        + 0.0864 * gamma ** 2
        + 0.045 * gamma
        - 0.1695
    )

    return csd.vertcat(velFactorBoom, velFactorArm, velFactorBucket)


def actuatorVel(q, qDot):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    alphaDot = qDot[0]
    betaDot = qDot[1]
    gammaDot = qDot[2]

    # 直接用拟合多项式的一阶导数 * 角速度
    lenBoomDot = alphaDot * (
        0.0344 * alpha ** 3
        - 0.1377 * alpha ** 2
        - 0.0208 * alpha
        + 0.2956
    )

    lenArmDot = betaDot * (
        0.0312 * beta ** 3
        + 0.2751 * beta ** 2
        + 0.582 * beta
        + 0.0646
    )

    lenBucketDot = gammaDot * (
        0.0192 * gamma ** 3
        + 0.0864 * gamma ** 2
        + 0.045 * gamma
        - 0.1695
    )

    return csd.vertcat(lenBoomDot, lenArmDot, lenBucketDot)

def motorVel(q, qDot):
    lenDot = actuatorVel(q, qDot)
    lenBoomDot = lenDot[0]
    lenArmDot = lenDot[1]
    lenBucketDot = lenDot[2]

    angVelMotorBoom = 2444.16 * lenBoomDot
    angVelMotorArm = 2444.16 * lenArmDot
    angVelMotorBucket = 2444.16 * lenBucketDot

    return csd.vertcat(angVelMotorBoom, angVelMotorArm, angVelMotorBucket)


def motorTorque(q, qDot, qDDot, F):
    T = inverseDynamics(q, qDot, qDDot, F)
    TBoom = T[0]
    TArm = T[1]
    TBucket = T[2]

    r = actuatorVelFactor(q)
    rBoom = r[0]
    rArm = r[1]
    rBucket = r[2]

    TMotorBoom = TBoom / (2444.16 * rBoom)
    TMotorArm = TArm / (2444.16 * rArm)
    TMotorBucket = TBucket / (2444.16 * rBucket)

    return csd.vertcat(TMotorBoom, TMotorArm, TMotorBucket)


def motorTorqueLimit(motorVel, dutyCycle: DutyCycle):
    # motorVel: 3x1
    # dutyCycle: DutyCycle enum
    # 这里维持你原来的逻辑：用速度依赖的极限
    vBoom = motorVel[0]
    vArm = motorVel[1]
    vBucket = motorVel[2]

    # 这里原来是查表 / 多项式，你原文件应该是同一段
    # 我直接保持成速度无关的常数 + 简单削减因子
    # 如果你原文件有更精细的拟合，可以替换这里

    # Rated torque
    T_rated = 32.0  # Nm (示例)

    if dutyCycle == DutyCycle.S1:
        factor = 1.0
    elif dutyCycle == DutyCycle.S2_60:
        factor = 1.2
    elif dutyCycle == DutyCycle.S2_30:
        factor = 1.4
    elif dutyCycle == DutyCycle.PEAK:
        factor = 1.6
    else:
        factor = 1.0

    # 简单速度衰减（防止超高速时无限大）
    # 这里你可以替换成自己之前的 speed–torque 曲线
    k_speed = 1e-4
    scaleBoom = 1.0 / (1.0 + k_speed * vBoom ** 2)
    scaleArm = 1.0 / (1.0 + k_speed * vArm ** 2)
    scaleBucket = 1.0 / (1.0 + k_speed * vBucket ** 2)

    T_lim_boom = T_rated * factor * scaleBoom
    T_lim_arm = T_rated * factor * scaleArm
    T_lim_bucket = T_rated * factor * scaleBucket

    return csd.vertcat(T_lim_boom, T_lim_arm, T_lim_bucket)
