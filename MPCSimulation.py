import casadi as csd
import numpy as np
import visualisation as vis
import excavatorModel as mod
from NLPSolver import NLP, Mode, Ts, integrator
from excavatorModel import DutyCycle

TMotor = 0.001 # Motor velocity command time interval
NMotor = int(Ts/TMotor) # Number of motor velocity commands per MPC time step
TSim = 5 # Simulation time period
NSim = int(TSim/Ts) # Number of time steps within simulation time period

x0 = csd.vertcat(1, -2.2, -1.8, 0, 0, 0)
poseDesired = csd.vertcat(3.2, -0.95737, -0.8)
qDesired = mod.inverseKinematics(poseDesired)

mode = Mode.LIFT
extF = 1000 # Magnitude of external force
dutyCycle = DutyCycle.S2_30

if mode != Mode.LIFT:
    extF = 0

# Simulate excavator arm state after one MPC time step (NMot motor commands)
def motorCommands(x, u):
    actuatorLenPrev = mod.actuatorLen(x[0:3]) # Current actuator length
    deltaActuatorLen = 0 # Total length change of linear actuator
    motorSpdFinal = 0 # Motor angular velocity at the end of the MPC time step

    for i in range(NMotor):
        xInterpolate = integrator(x, u, (i + 1)*TMotor)
        motorSpd = mod.motorVel(xInterpolate[0:3], xInterpolate[3:6])
        deltaMotorAng = TMotor*motorSpd
        deltaActuatorLen += deltaMotorAng/2444.16

        if i == NMotor - 1:
            motorSpdFinal = motorSpd

    actuatorLenNew = actuatorLenPrev + deltaActuatorLen
    actuatorVelNew = motorSpdFinal/2444.16
    qNew = mod.jointAngles(actuatorLenNew)
    qDotNew = mod.jointVel(qNew, actuatorVelNew)

    return csd.vertcat(qNew, qDotNew)

# Run the simulation and visualisation
xNext = x0

if mode == Mode.LIFT:
    vis.visualise(xNext, None, qDesired, 0, 0, [0, -extF])
else:
    vis.visualise(xNext, None, qDesired, 0, 0, None)

jointAng = x0[0:3]
jointVel = x0[3:6]
jointAcc = np.array([[0, 0, 0]]).T
motorTorque = np.array([[0, 0, 0]]).T
motorVel = np.array([[0, 0, 0]]).T
motorPower = np.array([[0, 0, 0]]).T

for k in range(NSim): # Run simulation for 50*Ts = 5 seconds
    opti = NLP(mode, extF, dutyCycle) # It seems to be necessary to reinstantiate the Opti stack, otherwise Casadi throws an overconstrained NLP error after about 25 MPC time steps
    sol = opti.solveNLP(xNext, poseDesired)

    xNext = motorCommands(sol.value(opti.x[:, 0]), sol.value(opti.u[:, 0]))

    if mode == Mode.LIFT:
        vis.visualise(xNext, None, qDesired, (k + 1)*Ts, k + 1, sol.value(opti.extForce[:, 0]))
    else:
        vis.visualise(xNext, None, qDesired, (k + 1)*Ts, k + 1, None)

    jointAng = np.hstack((jointAng, xNext[0:3]))
    jointVel = np.hstack((jointVel, xNext[3:6]))
    jointAcc = np.hstack((jointAcc, sol.value(opti.u[:, 0])[:, np.newaxis]))
    motorTorque = np.hstack((motorTorque, mod.motorTorque(xNext[0:3], xNext[3:6], jointAcc[:, -1], extF)))
    motorVel = np.hstack((motorVel, mod.motorVel(xNext[0:3], xNext[3:6])))
    motorPower = np.hstack((motorPower, np.abs(motorTorque[:,-1]*motorVel[:,-1])[:, np.newaxis]/1000))

    print(k)

vis.plotMotorOpPt(motorTorque, motorVel, dutyCycle)
vis.graph(0, TSim, Ts, "Joint Angles", "$\mathsf{q\ (rad)}$", x=jointAng)
vis.graph(0, TSim, Ts, "Joint Angular Velocities", "$\mathsf{\dot{q}\ (rad\ s^{-1})}$", x=jointVel)
vis.graph(0, TSim, Ts, "Joint Angular Accelerations", "$\mathsf{\ddot{q}\ (rad\ s^{-2})}$", u=jointAcc)
vis.graph(0, TSim, Ts, "Motor Torques", "Motor torque (Nm)", motorTorque=motorTorque)
vis.graph(0, TSim, Ts, "Motor Velocities", "Motor velocity ($\mathsf{rad\ s^{-1}}$)", motorVel=motorVel)
vis.graph(0, TSim, Ts, "Motor Powers", "Motor power (kW)", motorPower=motorPower)

vis.createVideo(0, NSim, "Excavator", int(1/Ts))

print("Done")