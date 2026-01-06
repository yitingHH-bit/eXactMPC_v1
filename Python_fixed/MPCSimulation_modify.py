import casadi as csd
import numpy as np

import visualisation as vis
import excavatorModel as mod
from NLPSolver import NLP, Mode, Ts, integrator
from excavatorModel import DutyCycle

print("MPCSimulation_modify: starting simulation...")

# ----------------------------------------------------------------------
# Simulation and motor-command parameters
# ----------------------------------------------------------------------
TMotor = 0.001  # Motor velocity command time interval
NMotor = int(Ts / TMotor)  # Number of motor velocity commands per MPC time step
TSim = 5.0  # Simulation time period [s]
NSim = int(TSim / Ts)  # Number of MPC steps within simulation time period

print(f"MPCSimulation_modify: Ts = {Ts}, TMotor = {TMotor}")
print(f"MPCSimulation_modify: NMotor = {NMotor}, TSim = {TSim}, NSim = {NSim}")

# ----------------------------------------------------------------------
# Initial state and desired pose (same as original script)
# State (full) x = [yaw, q1, q2, q3, yawdot, q1dot, q2dot, q3dot]
# ----------------------------------------------------------------------
# Initial yaw set to 0; original script only had the 3 joints
q_init = csd.vertcat(1, -2.2, -1.8)
x0 = csd.vertcat(0.0, q_init, 0.0, csd.DM.zeros(3, 1))
poseDesired = csd.vertcat(3.2, -0.95737, -0.8)
qDesired = mod.inverseKinematics(poseDesired)
target_xyz = csd.vertcat(poseDesired[0], 0.0, poseDesired[1])
target_phi = float(poseDesired[2])

# ----------------------------------------------------------------------
# Mode / external force / duty cycle
# ----------------------------------------------------------------------
mode = Mode.LIFT
extF = 1000  # Magnitude of external force
dutyCycle = DutyCycle.S2_30

if mode != Mode.LIFT:
    # For NO_LOAD / DIG we start with zero constant external force
    extF = 0

print(f"MPCSimulation_modify: mode = {mode}, extF = {extF}, dutyCycle = {dutyCycle}")

# ----------------------------------------------------------------------
# Simulate excavator arm state after one MPC time step (NMotor commands)
# ----------------------------------------------------------------------
def motorCommands(x_full, u_full):
    """
    Simulate the state over one MPC step Ts, using NMotor
    smaller integration steps of length TMotor and constant joint
    acceleration u. Yaw is propagated with a simple double integrator.

    Parameters
    ----------
    x_full : casadi.DM(8,1)
        Current full state [yaw, q; yawdot, qdot]
    u_full : casadi.DM(4,1)
        Current control [yawDDot, qDDot]

    Returns
    -------
    x_next : casadi.DM(8,1)
        State after Ts seconds.
    """
    yaw = x_full[0]
    q = x_full[1:4]
    yawdot = x_full[4]
    qdot = x_full[5:8]

    u_yaw = u_full[0]
    u_j = u_full[1:4]

    # Current actuator length
    actuatorLenPrev = mod.actuatorLen(q)

    deltaActuatorLen = 0  # Total length change of linear actuator
    motorSpdFinal = 0     # Motor angular velocity at end of MPC step

    for i in range(NMotor):
        # Interpolate state at time tau = (i+1)*TMotor within this MPC step
        tau = (i + 1) * TMotor
        xInterpolate = integrator(csd.vertcat(q, qdot), u_j, tau)

        # Motor speed from joint position/velocity at this sub-step
        motorSpd = mod.motorVel(xInterpolate[0:3], xInterpolate[3:6])

        # Motor angle increment and equivalent actuator extension
        deltaMotorAng = TMotor * motorSpd
        deltaActuatorLen += deltaMotorAng / 2444.16

        if i == NMotor - 1:
            motorSpdFinal = motorSpd

    # New actuator length and velocity
    actuatorLenNew = actuatorLenPrev + deltaActuatorLen
    actuatorVelNew = motorSpdFinal / 2444.16

    # Map actuator length/vel back to joint space
    qNew = mod.jointAngles(actuatorLenNew)
    qDotNew = mod.jointVel(qNew, actuatorVelNew)

    # Propagate yaw with double integrator over Ts
    yawNew = yaw + yawdot * Ts + 0.5 * u_yaw * Ts ** 2
    yawDotNew = yawdot + u_yaw * Ts

    return csd.vertcat(yawNew, qNew, yawDotNew, qDotNew)


# ----------------------------------------------------------------------
# Run the simulation loop
# ----------------------------------------------------------------------
xNext = x0

# Initial visualisation frame (k = 0)
q_vis = xNext[1:4]
qdot_vis = xNext[5:8]
x_vis = csd.vertcat(q_vis, qdot_vis)
if mode == Mode.LIFT:
    vis.visualise(x_vis, None, qDesired, 0.0, 0, [0, -extF])
else:
    vis.visualise(x_vis, None, qDesired, 0.0, 0, None)

# Histories (stored as NumPy arrays for plotting and GUI)
jointAng = np.array(q_vis, dtype=float).reshape(3, 1)          # shape (3, 1)
jointVel = np.array(qdot_vis, dtype=float).reshape(3, 1)       # shape (3, 1)
jointAcc = np.array([[0.0, 0.0, 0.0]]).T           # shape (3, 1)
motorTorque = np.array([[0.0, 0.0, 0.0]]).T        # shape (3, 1)
motorVel = np.array([[0.0, 0.0, 0.0]]).T           # shape (3, 1)
motorPower = np.array([[0.0, 0.0, 0.0]]).T         # shape (3, 1)

print("MPCSimulation_modify: entering main loop...")

for k in range(NSim):  # Run simulation for NSim * Ts = TSim seconds
    # Re-instantiate NLP each step (avoids over-constrained Opti issues)
    opti = NLP(mode, extF, dutyCycle)
    sol = opti.solveNLP(xNext, target_xyz, target_phi)

    # Optimal control at current step
    u0 = sol.value(opti.u[:, 0])  # shape (4,)
    u0_dm = csd.DM(u0)

    # Propagate state over one MPC step using motorCommands
    xNext = motorCommands(sol.value(opti.x[:, 0]), u0_dm)

    q_vis = xNext[1:4]
    qdot_vis = xNext[5:8]
    x_vis = csd.vertcat(q_vis, qdot_vis)

    # Visualise current configuration
    if mode == Mode.LIFT:
        # NLP no longer exposes extForce; visualize the constant downward force
        extF_vec = np.array([0.0, -extF], dtype=float)
        vis.visualise(x_vis, None, qDesired, (k + 1) * Ts, k + 1, extF_vec)
    else:
        vis.visualise(x_vis, None, qDesired, (k + 1) * Ts, k + 1, None)

    # ---- Log data for plotting / GUI ----
    qk = np.array(xNext[1:4], dtype=float).reshape(3, 1)
    qdotk = np.array(xNext[5:8], dtype=float).reshape(3, 1)
    uk = np.array(u0[1:], dtype=float).reshape(3, 1)

    jointAng = np.hstack((jointAng, qk))
    jointVel = np.hstack((jointVel, qdotk))
    jointAcc = np.hstack((jointAcc, uk))

    # Motor torque / velocity
    try:
        tau_dm = mod.motorTorque(qk[:, 0], qdotk[:, 0], uk[:, 0], extF)
        tau = np.array(tau_dm, dtype=float).reshape(3, 1)
    except AttributeError:
        # motorTorque not available in current excavatorModel; fall back to zeros
        tau = np.zeros((3, 1), dtype=float)
    w_dm = mod.motorVel(qk[:, 0], qdotk[:, 0])
    w = np.array(w_dm, dtype=float).reshape(3, 1)

    motorTorque = np.hstack((motorTorque, tau))
    motorVel = np.hstack((motorVel, w))

    # Instantaneous motor power in kW
    p = np.abs(tau[:, 0] * w[:, 0])[:, np.newaxis] / 1000.0
    motorPower = np.hstack((motorPower, p))

    if k % 5 == 0 or k == NSim - 1:
        print(f"MPCSimulation_modify: step {k+1}/{NSim} done")

print("MPCSimulation_modify: main loop finished, plotting & saving...")

# ----------------------------------------------------------------------
# Save MPC data for GUI playback
# ----------------------------------------------------------------------
time_vec = np.linspace(0.0, TSim, jointAng.shape[1])

np.savez(
    "mpc_results.npz",
    jointAng=jointAng,
    jointVel=jointVel,
    jointAcc=jointAcc,
    motorTorque=motorTorque,
    motorVel=motorVel,
    motorPower=motorPower,
    time=time_vec,
    Ts=Ts,
    TSim=TSim,
)

print("MPCSimulation_modify: saved MPC data to mpc_results.npz")

# ----------------------------------------------------------------------
# Plots and video using visualisation.py
# ----------------------------------------------------------------------
try:
    vis.plotMotorOpPt(motorTorque, motorVel, dutyCycle)
except Exception as e:
    print(f"MPCSimulation_modify: skipping motor torque plot ({e})")
vis.graph(0.0, TSim, Ts, "Joint Angles", r"$\mathsf{q\ (rad)}$", x=jointAng)
vis.graph(0.0, TSim, Ts, "Joint Angular Velocities", r"$\mathsf{\dot{q}\ (rad\ s^{-1})}$", x=jointVel)
vis.graph(0.0, TSim, Ts, "Joint Angular Accelerations", r"$\mathsf{\ddot{q}\ (rad\ s^{-2})}$", u=jointAcc)
vis.graph(0.0, TSim, Ts, "Motor Torques", "Motor torque (Nm)", motorTorque=motorTorque)
vis.graph(0.0, TSim, Ts, "Motor Velocities", r"Motor velocity ($\mathsf{rad\ s^{-1}}$)", motorVel=motorVel)
vis.graph(0.0, TSim, Ts, "Motor Powers", "Motor power (kW)", motorPower=motorPower)

print("MPCSimulation_modify: creating video...")
vis.createVideo(0, NSim, "Excavator", int(1.0 / Ts))

print("MPCSimulation_modify: Done.")
print(f"Results saved in folder: {vis.visFolder} and file mpc_results.npz")
