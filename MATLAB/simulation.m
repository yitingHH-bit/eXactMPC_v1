addpath(genpath(pwd));

% 构造挖掘机模型
e = excavatorModel(excavatorConstants, 0, 0, 0);

%{
Boom (0.905 - 1.305 m)   (-0.4737 -  1.0923 rad)
Arm  (1.046 - 1.556 m)   (-2.5848 - -0.5103 rad)
Bucket (0.84 - 1.26 m)   (-2.8659 -  0.7839 rad)
%}

% 先用给定的缸长设定一个姿态
[alpha, beta, gamma] = e.setLengths(1.2, 1.2, 1.2);
fprintf('Joint angles from setLengths: alpha=%.3f, beta=%.3f, gamma=%.3f rad\n', ...
        alpha, beta, gamma);

% ---------------------------------------------
%  原始 2D 可视化（和你之前一模一样）
% ---------------------------------------------
figure(1); clf;
e.visualise;
title('Original 2D excavator visualisation');

% ---------------------------------------------
%  NEW: 3D + yaw 可视化（对齐 Python GUI）
% ---------------------------------------------
% 选择几个 yaw 角看看末端如何绕世界 z 轴旋转
yawListDeg = [-120, -60, 0, 60, 120];
yawListRad = deg2rad(yawListDeg);

figure(2); clf; hold on; grid on;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Excavator 3D geometry with base yaw');
colors = lines(numel(yawListRad));

L = e.c.lenBA + e.c.lenAL + e.c.lenLM + 0.5;
for k = 1:numel(yawListRad)
    yaw = yawListRad(k);
    [ptsWorld, tipWorld] = e.forwardKinematics3D(yaw);

    plot3(ptsWorld(:,1), ptsWorld(:,2), ptsWorld(:,3), '-o', ...
          'Color', colors(k,:), 'LineWidth', 1.5);
    text(tipWorld(1), tipWorld(2), tipWorld(3), ...
         sprintf(' %d°', yawListDeg(k)), ...
         'Color', colors(k,:), 'FontSize', 8);
end

% Ground line
zG = e.c.yGround;
plot3([-L, L], [0, 0], [zG, zG], 'k--', 'LineWidth', 1.2);

axis equal;
view(35, 20);
hold off;

% ---------------------------------------------
%  其他测试（保持你之前的注释，方便手动打开）
% ---------------------------------------------
%[lenBoom, lenArm, lenBucket] = e.setAngles(0.5, -1, -1)
%[TBoom, TArm, TBucket] = e.inverseDynamics(1,2,3,4,5,6,[0;-1000])
%[FBoom, FArm, FBucket] = e.calcForces(3000, 2000, 1000)
