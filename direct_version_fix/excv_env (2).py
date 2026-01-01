from __future__ import annotations
import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat

# 尝试导入机器人配置
try:
    from isaaclab_assets.robots.masiV2 import MASIV2_CFG as CFG
except Exception:
    from isaaclab_assets.robots.masiV0 import MASI_CFG as CFG

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation around world Z) from wxyz quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

# -------------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------------

@configclass
class MasiEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10
    decimation = 2

    # Action space: 5 joints + 1 claw
    action_space = 6
    
    # Observation space: 
    # ee_quat(4) + log_quat(4) + relative_pos(3) + claw_state(1) = 12
    observation_space = 12
    state_space = 0

    # Claw positions (radians)
    claw_open_position = -50.0 * (3.14159 / 180.0)
    claw_closed_position = 29.0 * (3.14159 / 180.0)

    # Simulation settings
    sim = sim_utils.SimulationCfg(dt=1 / 60, render_interval=decimation // 2)
    
    # Robot Configuration
    robot: ArticulationCfg = CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Scene Configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)

    # --- Log Object Properties ---
    log_size = (0.05, 0.2, 0.05)
    log_color = (0.5, 0.1, 0.0)
    log_mass = 0.100
    log_z_position = 0.05

    # --- 关键修改：动态生成范围 (Polar Coordinate Settings) ---
    # 机械臂的物理限制：最小伸展距离和最大伸展距离
    # 请根据你的挖掘机实际尺寸调整这两个值，确保在机械臂"Range"内
    log_spawn_radius_min = 0.50  # 距离底座中心的最小半径 (米)
    log_spawn_radius_max = 0.65  # 距离底座中心的最大半径 (米)

    # 角度范围 (Curriculum Learning)
    # 初始只在正前方 +/- 10度生成
    log_angle_min_range = 0.17  # ~10 degrees
    # 最终在正前方 +/- 90度 (半圆) 生成，如果想360度，设为 3.14
    log_angle_max_range = 1.57  # ~90 degrees (Quarter circle left/right)
    
    # 课程学习进度：经过多少次 reset 后达到最大难度
    curriculum_steps_total = 10000 

    # Robot Randomization
    robot_joint_noise_scale = 0.01
    joint_action_scale = 0.05
    claw_action_filter_len = 5

# -------------------------------------------------------------------------
# Environment Logic
# -------------------------------------------------------------------------

class MasiEnv(DirectRLEnv):
    cfg: MasiEnvCfg

    def __init__(self, cfg: MasiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 记录总的训练/重置步数，用于课程学习
        self.total_resets_counter = 0

        # Find claw joints
        self.claw_joint_ids, _ = self.robot.find_joints("revolute_claw.*")

        # Claw state management
        self.claw_state = torch.zeros(self.num_envs, device=self.device)
        self._claw_action_hist = torch.zeros(
            self.num_envs, self.cfg.claw_action_filter_len, device=self.device
        )
        self._claw_action_hist_idx = 0

        # Configure robot entity
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["revolute_cabin", "revolute_lift", "revolute_tilt", "revolute_scoop", "revolute_gripper"],
            body_names=["ee"]
        )
        self.robot_entity_cfg.resolve(self.scene)

        self.control_joint_ids = self.robot_entity_cfg.joint_ids
        self.num_control_joints = len(self.control_joint_ids)

        # Joint limits
        joint_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.control_joint_limits = joint_limits[:, self.control_joint_ids, :]

        # Initialize joint targets
        self.joint_position_targets = self.robot.data.joint_pos[:, self.control_joint_ids].clone()

        print(f"[MasiEnv] Radius Range: {self.cfg.log_spawn_radius_min}m to {self.cfg.log_spawn_radius_max}m")

    def _setup_scene(self):
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Robot
        self.robot = Articulation(self.cfg.robot)

        # Log object
        log_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Log",
            spawn=sim_utils.CuboidCfg(
                size=self.cfg.log_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.log_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.log_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, self.cfg.log_z_position),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.logs = RigidObject(cfg=log_cfg)

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone and Add
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["logs"] = self.logs

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0).view(self.num_envs, -1)
        joint_actions = actions[:, :self.num_control_joints]
        claw_action = actions[:, -1]

        joint_pos = self.robot.data.joint_pos[:, self.control_joint_ids]
        target_joint_pos = joint_pos + joint_actions * self.cfg.joint_action_scale

        lower_limits = self.control_joint_limits[:, :, 0]
        upper_limits = self.control_joint_limits[:, :, 1]
        target_joint_pos = torch.max(torch.min(target_joint_pos, upper_limits), lower_limits)

        self.joint_position_targets = target_joint_pos

        # Update claw filter
        self._claw_action_hist[:, self._claw_action_hist_idx] = torch.sign(claw_action)
        self._claw_action_hist_idx = (self._claw_action_hist_idx + 1) % self.cfg.claw_action_filter_len
        claw_votes = self._claw_action_hist.sum(dim=1)
        self.claw_state = torch.where(
            claw_votes > 0.0, torch.ones_like(self.claw_state), torch.zeros_like(self.claw_state)
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_position_targets, joint_ids=self.control_joint_ids)
        
        claw_positions = torch.where(
            self.claw_state > 0.5,
            torch.full_like(self.claw_state, self.cfg.claw_open_position),
            torch.full_like(self.claw_state, self.cfg.claw_closed_position)
        )
        claw_positions_expanded = claw_positions.unsqueeze(1).expand(-1, len(self.claw_joint_ids))
        self.robot.set_joint_position_target(claw_positions_expanded, joint_ids=self.claw_joint_ids)

    def _get_observations(self) -> dict:
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], :7]
        log_pose_w = self.logs.data.root_state_w[:, :7]

        ee_pos_local = ee_pose_w[:, :3] - self.scene.env_origins
        log_pos_local = log_pose_w[:, :3] - self.scene.env_origins

        ee_quat = ee_pose_w[:, 3:7]
        log_quat = log_pose_w[:, 3:7]

        relative_pos = ee_pos_local - log_pos_local
        # Normalize relative pos roughly for observation
        relative_pos_scaled = torch.clamp(relative_pos / 1.0, -1.0, 1.0) 

        claw_scaled = self.claw_state.unsqueeze(1) * 2.0 - 1.0

        obs = torch.cat([ee_quat, log_quat, relative_pos_scaled, claw_scaled], dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Same reward function as before
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], :7]
        log_pose_w = self.logs.data.root_state_w[:, :7]

        ee_pos_local = ee_pose_w[:, :3] - self.scene.env_origins
        log_pos_local = log_pose_w[:, :3] - self.scene.env_origins

        relative_pos = ee_pos_local - log_pos_local
        ee_quat = ee_pose_w[:, 3:7]
        log_quat = log_pose_w[:, 3:7]

        distance = torch.norm(relative_pos, dim=-1)
        distance_reward = -distance * 5.0

        ee_yaw = quat_to_yaw(ee_quat)
        log_yaw = quat_to_yaw(log_quat)
        yaw_diff = torch.abs(wrap_to_pi(ee_yaw - log_yaw))
        symmetric_yaw_diff = torch.minimum(yaw_diff, torch.abs(torch.pi - yaw_diff))
        yaw_alignment = 1.0 - torch.clamp(symmetric_yaw_diff / (0.5 * torch.pi), 0.0, 1.0)
        yaw_reward = (yaw_alignment ** 4) * 0.5

        ee_rot_mats = matrix_from_quat(ee_quat)
        ee_z_axis = ee_rot_mats[:, :, 2]
        vertical_alignment = torch.clamp(ee_z_axis[:, 2], 0.0, 1.0)
        vertical_reward = (vertical_alignment ** 4) * 1.0

        threshold = 0.05
        open_far_mask = (distance > threshold).float()
        close_near_mask = (distance < threshold).float()
        grasp_reward = (open_far_mask * self.claw_state * 0.2) + (close_near_mask * (1.0 - self.claw_state) * 0.5)

        lifted_height = torch.clamp(log_pos_local[:, 2] - self.cfg.log_z_position, min=0.0)
        lift_reward = lifted_height * 500.0

        rewards = distance_reward + yaw_reward + vertical_reward + grasp_reward + lift_reward
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_robot_joints(self, env_ids: torch.Tensor):
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        dof_lower = joint_pos_limits[..., 0]
        dof_upper = joint_pos_limits[..., 1]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        
        # Add noise
        delta_max = dof_upper[env_ids] - joint_pos
        delta_min = dof_lower[env_ids] - joint_pos
        noise = torch.rand(len(env_ids), self.robot.num_joints, device=self.device) * 2.0 - 1.0
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * noise
        
        joint_pos_rand = joint_pos + self.cfg.robot_joint_noise_scale * rand_delta
        joint_pos_rand = torch.clamp(joint_pos_rand, dof_lower[env_ids], dof_upper[env_ids])
        
        self.robot.set_joint_position_target(joint_pos_rand, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos_rand, torch.zeros_like(joint_pos_rand), env_ids=env_ids)

        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)

    def _reset_log_positions(self, env_ids: torch.Tensor):
        """
        核心逻辑：使用极坐标系 (Polar Coordinates) 围绕挖掘机更新目标。
        确保目标始终在 (min_radius, max_radius) 之间。
        """
        num_to_reset = len(env_ids)
        
        # --- 1. 计算当前的难度 (Curriculum) ---
        # 随着 reset 次数增加，难度系数从 0.0 增加到 1.0
        self.total_resets_counter += num_to_reset
        difficulty = min(self.total_resets_counter / self.cfg.curriculum_steps_total, 1.0)
        
        # 根据难度计算当前的角度范围
        # 开始时很窄 (min_range)，最后变宽 (max_range)
        current_cone_half_angle = (
            self.cfg.log_angle_min_range + 
            difficulty * (self.cfg.log_angle_max_range - self.cfg.log_angle_min_range)
        )

        # --- 2. 极坐标采样 (Polar Sampling) ---
        
        # A. 随机半径 r: 在 min_radius 和 max_radius 之间均匀分布
        # 这保证了 log 永远不会超出机械臂的触达范围，也不会离得太近撞到车身
        r = (self.cfg.log_spawn_radius_min + 
             torch.rand(num_to_reset, device=self.device) * (self.cfg.log_spawn_radius_max - self.cfg.log_spawn_radius_min))
        
        # B. 随机角度 theta: 在 [-current_cone, +current_cone] 之间
        theta = (torch.rand(num_to_reset, device=self.device) * 2.0 - 1.0) * current_cone_half_angle
        
        # --- 3. 转换回笛卡尔坐标 (Cartesian) ---
        log_x = r * torch.cos(theta)
        log_y = r * torch.sin(theta)
        log_z = torch.full((num_to_reset,), self.cfg.log_z_position, device=self.device)
        
        # 组合本地坐标
        pos_local = torch.stack([log_x, log_y, log_z], dim=1)
        
        # 加上环境原点偏移，得到世界坐标
        pos_world = pos_local + self.scene.env_origins[env_ids]
        
        # --- 4. 随机旋转 Log 自身 ---
        # 让木头自身也有随机的 Yaw 旋转，增加抓取难度
        rand_yaw = torch.rand(num_to_reset, device=self.device) * 2 * math.pi
        quat = quat_from_euler_xyz(
            torch.zeros(num_to_reset, device=self.device),
            torch.zeros(num_to_reset, device=self.device),
            rand_yaw
        )
        
        # --- 5. 写入仿真 ---
        pose = torch.cat([pos_world, quat], dim=1)
        self.logs.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.logs.write_root_velocity_to_sim(torch.zeros(num_to_reset, 6, device=self.device), env_ids=env_ids)

    def _sync_joint_targets(self, env_ids: torch.Tensor):
        self.robot.update(self.sim.get_physics_dt())
        curr_pos = self.robot.data.joint_pos[env_ids][:, self.control_joint_ids]
        lower = self.control_joint_limits[env_ids, :, 0]
        upper = self.control_joint_limits[env_ids, :, 1]
        self.joint_position_targets[env_ids] = torch.max(torch.min(curr_pos, upper), lower)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.claw_state[env_ids] = 0.0
        self._reset_robot_joints(env_ids)
        self._reset_log_positions(env_ids) # 这里调用新的极坐标更新逻辑
        self._sync_joint_targets(env_ids)