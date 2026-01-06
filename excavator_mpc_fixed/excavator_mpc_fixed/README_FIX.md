# Excavator MPC 修复说明（URDF: model2_simple / robot.urdf）

你提到的 3 个问题，我在代码里做了对应修复：

## 1) 设置目标位置偶发报错（reshape size=1 / IK 报错）
**根因**主要有两类：

- **poseDesired 的类型/形状不稳定**：旧版本 GUI/脚本把 `poseDesired` 设成 CasADi `DM/MX`，传到 `NLPSolver.solveNLP()` 里再 `np.asarray(...).reshape(3)` 时，某些情况下会变成 *object array*（size=1），就会出现你看到的 reshape 报错。
- **IK 数学不稳**：旧 `inverseKinematics()` 里 `sqrt(1-cos^2)` 在目标点接近不可达/数值误差时会出现负数开方。

**修复点**：
- `mpc_realtime_gui.py`：任何时候都把目标写成 **`numpy (3,)`**，并且每个 MPC step 再做一次 `np.asarray(...).reshape(3,)` 的双保险。
- `NLPSolver.py`：`solveNLP()` 里先 `reshape(-1)` 再检查元素数量，避免 CasADi 变成 size=1 的坑。
- `excavatorModel.py`：IK 对 `cos(beta)` 做 clamp，避免负数开方；IK 失败只提示，不会把程序直接打崩（GUI里也是如此）。

## 2) 你说的 Mimic（夹爪/机械臂）长度与 URDF 不匹配？
你给的 URDF 里：
- `gripper_claw_1_FixedJoint`、`gripper_claw_2_FixedJoint`、`gripper_frame_FixedJoint` **全部是 fixed joint**
- 文件里 **没有 `<mimic .../>` 标签**

所以严格来说：**URDF 当前没有任何“可动夹爪关节”，也就不存在 mimic 关系是否匹配**。

如果你想要“夹爪开合 + mimic”，需要把 claw 的 fixed joint 改成 `revolute`（或 prismatic），然后在其中一个 joint 上加 `<mimic joint="另一个关节" multiplier="-1"/>`（通常是一正一负对称）。

## 3) URDF 几何长度/末端点（TCP）与代码不匹配
之前 `excavatorModel.py` 里 **`inverseKinematics()` 被定义了两次**：
- 第一版是你新 URDF 的简化 L1/L2/L3
- 但后面又被“旧模型（lenBA/lenAL/lenLM）”覆盖了

这会导致：FK/IK 用的几何长度不一致，目标点有时会 IK 失败或解很离谱。

**修复点**：
- 删除重复定义，统一使用 **URDF-fitted** 的平面运动学。
- 进一步提升拟合精度：URDF 的 joint-to-joint 平移向量并不沿 +X 轴（例如 lift->tilt 同时有 x 和 z），因此我引入：
  - `OFF1/OFF2`（固定方向偏置）
  - `TCP_DX/TCP_DZ`（tool_body 到“夹爪中心”的固定偏移）

这样 FK/IK 更接近你的 URDF。

---

# 你现在应该运行哪个文件？

✅ 推荐直接跑：

```bash
python mpc_realtime_gui.py
```

旧的 `mpc_realtime_gui v1.py / v2.py / ok.py` 和 `MPCSimulation*.py` 有的仍然是早期实验稿（3-DOF state、CasADi 直接传参等），不建议再用。

