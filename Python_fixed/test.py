from excavatorModel import forward_kinematics_planar, forward_kinematics_3d

q = [0.4, 0.5, -0.3, 0.2]  # [yaw, boom, arm, tool]

planar = forward_kinematics_planar(q[1:])
p3d = forward_kinematics_3d(q)

print("Planar [x,z,phi]:", planar)
print("3D position [x,y,z]:", p3d)
