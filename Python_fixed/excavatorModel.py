from __future__ import annotations

import enum
from dataclasses import dataclass

import casadi as csd
import numpy as np

import excavatorConstants as C

# =============================================================================
# Continuous-time dynamics
# States:
#   q = [yaw, alpha, beta, gamma]
#   qDot = [yawDot, alphaDot, betaDot, gammaDot]
# Inputs:
#   u = [yawDDot, alphaDDot, betaDDot, gammaDDot]
# NOTE: This file is used by NLPSolver (MPC cost uses forwardKinematics()).
# =============================================================================


class DutyCycle(enum.Enum):
    S1_00 = 1.00
    S2_30 = 2.30
    S2_80 = 2.80


# -----------------------------------------------------------------------------
# Kinematics (Planar X-Z; yaw handled in GUI)
# -----------------------------------------------------------------------------

def forwardKinematicsPlanar(q):
    """Planar end-effector pose from joint angles.

    Conventions (matched to URDF joint axis = +Y, right-hand rule):
    - A positive joint angle rotates the link vector in the X–Z plane by **-angle**
      (because a +Y rotation maps x' = cos a * x + sin a * z, z' = -sin a * x + cos a * z).

    The URDF-fitted constants in excavatorConstants.py define:
      - L1, OFF1: lift_boom -> tilt joint vector length and its zero-angle direction
      - L2, OFF2: tilt_boom -> scoop/tool joint vector length and its zero-angle direction
      - TCP_DX, TCP_DZ: tool_body -> TCP offset expressed in tool_body frame

    Returns:
        casadi.DM (3,1): [x_tcp, z_tcp, phi] where phi = q1+q2+q3.
    """
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    # Link absolute directions (in world X–Z plane) at current angles
    th1 = C.OFF1 - alpha
    th2 = C.OFF2 - (alpha + beta)

    x_tool = C.L1 * csd.cos(th1) + C.L2 * csd.cos(th2)
    z_tool = C.L1 * csd.sin(th1) + C.L2 * csd.sin(th2)

    phi = alpha + beta + gamma

    # TCP offset rotated by -phi in the X–Z plane
    c = csd.cos(phi)
    s = csd.sin(phi)
    # rotation by -phi: [x;z] = [ c  s; -s  c ] [dx;dz]
    x_tcp = x_tool + (C.TCP_DX * c + C.TCP_DZ * s)
    z_tcp = z_tool + (-C.TCP_DX * s + C.TCP_DZ * c)

    return csd.vertcat(x_tcp, z_tcp, phi)


def forwardKinematics(q_joints):
    """3-DoF planar FK wrapper used by NLPSolver."""
    return forwardKinematicsPlanar(q_joints)


def inverseKinematics(pose, elbow_up=False):
    """Inverse kinematics consistent with forwardKinematicsPlanar().

    Args:
        pose: casadi DM/MX or array-like (3,) -> [x_tcp, z_tcp, phi]
        elbow_up: choose the elbow configuration (False = elbow-down)

    Returns:
        casadi.DM (3,1): [alpha, beta, gamma]
    """
    x = pose[0]
    z = pose[1]
    phi = pose[2]

    # Remove TCP offset (rotated by -phi)
    c = csd.cos(phi)
    s = csd.sin(phi)
    # [dx_w; dz_w] = Rot(-phi) [TCP_DX; TCP_DZ] = [ c  s; -s  c ] [dx;dz]
    dx_w = C.TCP_DX * c + C.TCP_DZ * s
    dz_w = -C.TCP_DX * s + C.TCP_DZ * c

    x_tool = x - dx_w
    z_tool = z - dz_w

    r2 = x_tool**2 + z_tool**2

    # Relative angle between links in the standard 2-link planar form
    cos_delta = (r2 - C.L1**2 - C.L2**2) / (2.0 * C.L1 * C.L2)
    cos_delta = csd.fmin(1.0, csd.fmax(-1.0, cos_delta))

    sin_delta_mag = csd.sqrt(csd.fmax(0.0, 1.0 - cos_delta**2))
    sin_delta = -sin_delta_mag if elbow_up else sin_delta_mag
    delta = csd.atan2(sin_delta, cos_delta)

    # Absolute direction of link1 in the X–Z plane
    k1 = C.L1 + C.L2 * cos_delta
    k2 = C.L2 * sin_delta
    theta1 = csd.atan2(z_tool, x_tool) - csd.atan2(k2, k1)
    theta2 = theta1 + delta

    # Map back to (alpha, beta) using our conventions:
    # theta1 = OFF1 - alpha  => alpha = OFF1 - theta1
    # delta = theta2 - theta1 = (OFF2 - (alpha+beta)) - (OFF1 - alpha) = (OFF2 - OFF1) - beta
    # => beta = (OFF2 - OFF1) - delta
    alpha = C.OFF1 - theta1
    beta = (C.OFF2 - C.OFF1) - delta
    gamma = phi - alpha - beta

    return csd.vertcat(alpha, beta, gamma)


# -----------------------------------------------------------------------------
# Original dynamic helpers (kept as-is; MPC integrator in NLPSolver is double integrator)
# -----------------------------------------------------------------------------

def actuatorLen(q_joints):
    """
    q_joints is [alpha, beta, gamma]
    returns actuatorLen = [lenBoom, lenArm, lenBucket]
    """
    alpha = q_joints[0]
    beta = q_joints[1]
    gamma = q_joints[2]

    lenBoom = 0.0086 * alpha**4 - 0.0459 * alpha**3 - 0.0104 * alpha**2 + 0.2956 * alpha + 1.042
    lenArm = 0.0078 * beta**4 + 0.0917 * beta**3 + 0.2910 * beta**2 + 0.0646 * beta + 1.0149
    lenBucket = 0.0048 * gamma**4 + 0.0288 * gamma**3 + 0.0225 * gamma**2 - 0.1695 * gamma + 0.9434

    return csd.vertcat(lenBoom, lenArm, lenBucket)


def jointAngles(actuatorLen):
    """
    actuatorLen = [lenBoom, lenArm, lenBucket]
    returns q_joints = [alpha, beta, gamma]
    """
    lenBoom = actuatorLen[0]
    lenArm = actuatorLen[1]
    lenBucket = actuatorLen[2]

    alpha = 0.5273 * lenBoom**4 - 3.3773 * lenBoom**3 + 7.7904 * lenBoom**2 - 7.9700 * lenBoom + 3.0559
    beta = 0.1591 * lenArm**4 - 0.6482 * lenArm**3 - 0.1315 * lenArm**2 + 1.6630 * lenArm - 1.6158
    gamma = -1.0394 * lenBucket**4 + 3.7311 * lenBucket**3 - 4.7254 * lenBucket**2 + 1.5662 * lenBucket - 0.3458

    return csd.vertcat(alpha, beta, gamma)


def motorVel(q_joints, qDot_joints):
    """
    q_joints = [alpha, beta, gamma]
    qDot_joints = [alphaDot, betaDot, gammaDot]
    returns motor speed [motorSpdBoom, motorSpdArm, motorSpdBucket]
    """
    alpha = q_joints[0]
    beta = q_joints[1]
    gamma = q_joints[2]

    alphaDot = qDot_joints[0]
    betaDot = qDot_joints[1]
    gammaDot = qDot_joints[2]

    # Jacobian (poly derivatives)
    dLenBoom = (0.0086 * 4 * alpha**3) - (0.0459 * 3 * alpha**2) - (0.0104 * 2 * alpha) + 0.2956
    dLenArm = (0.0078 * 4 * beta**3) + (0.0917 * 3 * beta**2) + (0.2910 * 2 * beta) + 0.0646
    dLenBucket = (0.0048 * 4 * gamma**3) + (0.0288 * 3 * gamma**2) + (0.0225 * 2 * gamma) - 0.1695

    actuatorVelBoom = dLenBoom * alphaDot
    actuatorVelArm = dLenArm * betaDot
    actuatorVelBucket = dLenBucket * gammaDot

    motorSpdBoom = actuatorVelBoom * 2444.16
    motorSpdArm = actuatorVelArm * 2444.16
    motorSpdBucket = actuatorVelBucket * 2444.16

    return csd.vertcat(motorSpdBoom, motorSpdArm, motorSpdBucket)


def jointVel(q_joints, actuatorVel):
    """
    actuatorVel = [lenBoomDot, lenArmDot, lenBucketDot]
    returns qDot_joints
    """
    alpha = q_joints[0]
    beta = q_joints[1]
    gamma = q_joints[2]

    lenBoomDot = actuatorVel[0]
    lenArmDot = actuatorVel[1]
    lenBucketDot = actuatorVel[2]

    dLenBoom = (0.0086 * 4 * alpha**3) - (0.0459 * 3 * alpha**2) - (0.0104 * 2 * alpha) + 0.2956
    dLenArm = (0.0078 * 4 * beta**3) + (0.0917 * 3 * beta**2) + (0.2910 * 2 * beta) + 0.0646
    dLenBucket = (0.0048 * 4 * gamma**3) + (0.0288 * 3 * gamma**2) + (0.0225 * 2 * gamma) - 0.1695

    alphaDot = lenBoomDot / dLenBoom
    betaDot = lenArmDot / dLenArm
    gammaDot = lenBucketDot / dLenBucket

    return csd.vertcat(alphaDot, betaDot, gammaDot)


# ----------------------------------------------------------------------
# Placeholder motor torque helpers (original dynamics not available here)
# ----------------------------------------------------------------------
def motorTorque(q_joints, qDot_joints, qDDot_joints, ext_force=0.0):
    """
    Lightweight placeholder for motor torque computation.

    Returns zeros to keep downstream logging/plots running when detailed
    actuator dynamics are not implemented in this simplified model.
    """
    try:
        return csd.vertcat(0.0, 0.0, 0.0)
    except Exception:
        return csd.DM([0.0, 0.0, 0.0])


def motorTorqueLimit(angVel, dutyCycle):
    """
    Simplified constant torque limit curve used for plotting.

    Args:
        angVel: angular velocity (scalar)
        dutyCycle: DutyCycle enum or float (scale factor)
    """
    duty = float(dutyCycle.value) if isinstance(dutyCycle, DutyCycle) else float(dutyCycle)
    base_limit = 1500.0  # Nm nominal placeholder
    return base_limit / max(duty, 1e-3)
