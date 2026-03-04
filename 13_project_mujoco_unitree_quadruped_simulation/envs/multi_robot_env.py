"""
Coklu Robot Quadruped Ortami / Multi-Robot Quadruped Environment

Birden fazla Unitree Go2 tarzinda quadruped robotun ayni ortamda
formation halinde hareket etmesini saglayan ortam.

Multi-robot environment where multiple Unitree Go2-style quadruped
robots move in formation in a shared simulation.

Her robot icin gozlem: kendi durumu (39) + diger robotlarin goreceli pozisyonlari
Each robot observation: own state (39) + relative positions of other robots
"""

import numpy as np
import mujoco
import tempfile
import os
from typing import List, Tuple, Optional
from pathlib import Path


class MultiRobotQuadrupedEnv:
    """
    Coklu quadruped robot ortami.
    Multi-robot quadruped environment using raw MuJoCo bindings.

    N robot ayni simülasyonda formation halinde hareket eder.
    N robots move in formation within the same simulation.
    """

    LEG_NAMES = ["FL", "FR", "BL", "BR"]
    JOINTS_PER_LEG = 3
    JOINTS_PER_ROBOT = 12

    def __init__(
        self,
        num_robots: int = 2,
        formation_type: str = "line",
        formation_spacing: float = 1.5,
        frame_skip: int = 10,
        forward_reward_weight: float = 1.0,
        formation_reward_weight: float = 2.0,
        ctrl_cost_weight: float = 0.05,
        healthy_reward: float = 1.0,
        healthy_z_range: Tuple[float, float] = (0.15, 0.55),
        max_episode_steps: int = 1000,
    ):
        self.num_robots = num_robots
        self.formation_type = formation_type
        self.formation_spacing = formation_spacing
        self.frame_skip = frame_skip
        self._forward_reward_weight = forward_reward_weight
        self._formation_reward_weight = formation_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Hedef formation pozisyonlari / Target formation offsets
        self._formation_offsets = self._compute_formation_offsets()

        # Multi-robot XML olustur ve yukle / Generate and load multi-robot XML
        xml_string = self._generate_multi_robot_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Her robot icin qpos/qvel indeks araliklari / Index ranges per robot
        # Her robot: 7 (free joint) + 12 (hinge joints) = 19 qpos
        # Her robot: 6 (free joint) + 12 (hinge joints) = 18 qvel
        self._qpos_per_robot = 19
        self._qvel_per_robot = 18
        self._act_per_robot = 12

        # Gozlem boyutlari / Observation dimensions
        self.robot_obs_dim = 39  # tek robot gozlemi
        self.relative_obs_dim = 6 * (num_robots - 1)  # goreceli pos + vel
        self.obs_dim_per_robot = self.robot_obs_dim + self.relative_obs_dim
        self.action_dim_per_robot = 12

    def _compute_formation_offsets(self) -> np.ndarray:
        """
        Hedef formation offset'lerini hesapla.
        Compute target formation offsets relative to centroid.
        """
        offsets = np.zeros((self.num_robots, 2))  # x, y offsets
        if self.formation_type == "line":
            for i in range(self.num_robots):
                offsets[i, 1] = (i - (self.num_robots - 1) / 2) * self.formation_spacing
        elif self.formation_type == "triangle" and self.num_robots >= 3:
            offsets[0] = [self.formation_spacing / 2, 0]
            offsets[1] = [-self.formation_spacing / 2, self.formation_spacing / 2]
            if self.num_robots > 2:
                offsets[2] = [-self.formation_spacing / 2, -self.formation_spacing / 2]
        return offsets

    def _generate_multi_robot_xml(self) -> str:
        """
        Birden fazla robot iceren MuJoCo XML'i programatik olarak olustur.
        Programmatically generate MuJoCo XML with multiple robots.
        """
        # Tek robotun bacak XML sablonu
        def leg_xml(prefix: str, pos: str, abd_axis: str) -> str:
            return f"""
      <body name="{prefix}_abd_link" pos="{pos}">
        <joint name="{prefix}_abd_joint" type="hinge" axis="1 0 0" range="-25 25"/>
        <geom name="{prefix}_abd_geom" type="capsule" fromto="0 0 0 {abd_axis}" size="0.02" rgba="0.3 0.3 0.8 1"/>
        <body name="{prefix}_hip_link" pos="{abd_axis}">
          <joint name="{prefix}_hip_joint" type="hinge" axis="0 1 0" range="-60 60"/>
          <geom name="{prefix}_upper_leg" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" rgba="0.4 0.4 0.9 1"/>
          <body name="{prefix}_knee_link" pos="0 0 -0.2">
            <joint name="{prefix}_knee_joint" type="hinge" axis="0 1 0" range="-145 -45"/>
            <geom name="{prefix}_lower_leg" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015" rgba="0.35 0.35 0.85 1"/>
            <geom name="{prefix}_foot" type="sphere" pos="0 0 -0.2" size="0.025" rgba="0.6 0.6 0.6 1" conaffinity="1"/>
            <site name="{prefix}_foot_site" pos="0 0 -0.2" size="0.025"/>
          </body>
        </body>
      </body>"""

        def robot_body_xml(robot_id: int, x_offset: float, y_offset: float) -> str:
            p = f"r{robot_id}"
            color = ["0.2 0.2 0.2 1", "0.6 0.2 0.2 1", "0.2 0.6 0.2 1"][robot_id % 3]
            body = f"""
    <body name="{p}_torso" pos="{x_offset} {y_offset} 0.35">
      <joint name="{p}_root" type="free" armature="0" damping="0" limited="false"/>
      <geom name="{p}_torso_geom" type="box" size="0.18 0.047 0.04" rgba="{color}" mass="5.0"/>
      <site name="{p}_torso_imu" pos="0 0 0" size="0.01"/>
      {leg_xml(f"{p}_fl", "0.18 0.047 0", "0 0.04 0")}
      {leg_xml(f"{p}_fr", "0.18 -0.047 0", "0 -0.04 0")}
      {leg_xml(f"{p}_bl", "-0.18 0.047 0", "0 0.04 0")}
      {leg_xml(f"{p}_br", "-0.18 -0.047 0", "0 -0.04 0")}
    </body>"""
            return body

        def robot_actuators_xml(robot_id: int) -> str:
            p = f"r{robot_id}"
            lines = []
            for leg in ["fl", "fr", "bl", "br"]:
                for joint in ["abd", "hip", "knee"]:
                    gear = "35" if joint == "knee" else "25"
                    lines.append(
                        f'    <motor name="{p}_{leg}_{joint}_motor" '
                        f'joint="{p}_{leg}_{joint}_joint" gear="{gear}"/>'
                    )
            return "\n".join(lines)

        def robot_sensors_xml(robot_id: int) -> str:
            p = f"r{robot_id}"
            lines = []
            for leg in ["fl", "fr", "bl", "br"]:
                lines.append(f'    <touch name="{p}_{leg}_foot_touch" site="{p}_{leg}_foot_site"/>')
            lines.append(f'    <gyro name="{p}_torso_gyro" site="{p}_torso_imu"/>')
            lines.append(f'    <accelerometer name="{p}_torso_accel" site="{p}_torso_imu"/>')
            return "\n".join(lines)

        # Robotlarin baslangic pozisyonlari / Initial positions
        bodies = ""
        actuators = ""
        sensors = ""
        for i in range(self.num_robots):
            x_off = 0.0
            y_off = self._formation_offsets[i, 1] if i < len(self._formation_offsets) else i * self.formation_spacing
            bodies += robot_body_xml(i, x_off, y_off)
            actuators += robot_actuators_xml(i) + "\n"
            sensors += robot_sensors_xml(i) + "\n"

        xml = f"""<mujoco model="multi_robot_quadruped">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint armature="0.01" damping="0.5" limited="true"/>
    <geom conaffinity="0" condim="3" density="1000" friction="1.0 0.5 0.5" margin="0.01" rgba="0.15 0.15 0.15 1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="0.6 0.8 1.0" rgb2="0.1 0.1 0.3" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0.2 0.3 0.4" rgb2="0.8 0.85 0.9" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
    {bodies}
  </worldbody>
  <actuator>
{actuators}  </actuator>
  <sensor>
{sensors}  </sensor>
</mujoco>"""
        return xml

    # =========================================================================
    # Ortam Arayuzu / Environment Interface
    # =========================================================================

    def reset(self, seed: int = None) -> List[np.ndarray]:
        """Ortami sifirla, her robot icin gozlem dondur / Reset, return obs per robot."""
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Kucuk rastgele gurultu ekle / Add small random noise
        noise_scale = 0.05
        self.data.qpos[:] += np.random.uniform(
            -noise_scale, noise_scale, size=self.data.qpos.shape
        )
        self.data.qvel[:] += noise_scale * np.random.randn(*self.data.qvel.shape)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_all_observations()

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Tum robotlar icin aksiyonlari uygula.
        Apply actions for all robots.

        Args:
            actions: Her robot icin (12,) boyutlu aksiyon listesi

        Returns:
            observations, rewards, done, info
        """
        # Onceki pozisyonlar / Previous positions
        prev_positions = self._get_robot_positions()

        # Tum aksiyonlari birlestir / Concatenate all actions
        full_action = np.concatenate(actions)
        self.data.ctrl[:] = full_action

        # Simulasyon adimi / Simulation step
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Sonraki pozisyonlar / Current positions
        curr_positions = self._get_robot_positions()

        # Gozlemler / Observations
        observations = self._get_all_observations()

        # Odulleri hesapla / Compute rewards
        rewards, reward_info = self._compute_rewards(
            prev_positions, curr_positions, actions
        )

        # Bitirme kontrolu / Termination check
        done = self._check_termination() or self._step_count >= self.max_episode_steps

        info = {
            "robot_positions": curr_positions,
            "formation_error": reward_info["formation_error"],
            "avg_forward_vel": reward_info["avg_forward_vel"],
            "step_count": self._step_count,
        }

        return observations, rewards, done, info

    # =========================================================================
    # Ic Metotlar / Internal Methods
    # =========================================================================

    def _get_robot_qpos(self, robot_idx: int) -> np.ndarray:
        """Robot'un qpos degerlerini al / Get robot's qpos values."""
        start = robot_idx * self._qpos_per_robot
        return self.data.qpos[start:start + self._qpos_per_robot]

    def _get_robot_qvel(self, robot_idx: int) -> np.ndarray:
        """Robot'un qvel degerlerini al / Get robot's qvel values."""
        start = robot_idx * self._qvel_per_robot
        return self.data.qvel[start:start + self._qvel_per_robot]

    def _get_robot_positions(self) -> np.ndarray:
        """Tum robotlarin x,y,z pozisyonlari / All robots' x,y,z positions."""
        positions = np.zeros((self.num_robots, 3))
        for i in range(self.num_robots):
            qpos = self._get_robot_qpos(i)
            positions[i] = qpos[:3]
        return positions

    def _get_robot_obs(self, robot_idx: int) -> np.ndarray:
        """
        Tek robot icin 39 boyutlu gozlem olustur.
        Construct 39-dim observation for a single robot.
        """
        qpos = self._get_robot_qpos(robot_idx)
        qvel = self._get_robot_qvel(robot_idx)

        z_pos = qpos[2:3]           # 1
        orientation = qpos[3:7]     # 4
        joint_angles = qpos[7:]     # 12
        body_vel = qvel[:6]         # 6
        joint_vel = qvel[6:]        # 12

        # Ayak temasi (basitlestirilmis: her zaman sifir)
        # Foot contacts (simplified: always zero, full contact detection requires more work)
        foot_contacts = np.zeros(4)

        return np.concatenate([
            z_pos, orientation, joint_angles,
            body_vel, joint_vel, foot_contacts
        ])

    def _get_all_observations(self) -> List[np.ndarray]:
        """Tum robotlar icin gozlemleri olustur (kendi durumu + goreceli pozisyonlar)."""
        robot_obs_list = [self._get_robot_obs(i) for i in range(self.num_robots)]
        positions = self._get_robot_positions()

        observations = []
        for i in range(self.num_robots):
            # Kendi gozlemi / Own observation
            own_obs = robot_obs_list[i]

            # Diger robotlarin goreceli durumu / Relative state of other robots
            relative_parts = []
            for j in range(self.num_robots):
                if j == i:
                    continue
                # Goreceli pozisyon ve hiz / Relative position and velocity
                rel_pos = positions[j] - positions[i]
                qvel_j = self._get_robot_qvel(j)
                rel_vel = qvel_j[:3] - self._get_robot_qvel(i)[:3]
                relative_parts.append(np.concatenate([rel_pos, rel_vel]))

            if relative_parts:
                relative_obs = np.concatenate(relative_parts)
            else:
                relative_obs = np.array([])

            observations.append(np.concatenate([own_obs, relative_obs]))

        return observations

    def _compute_rewards(
        self,
        prev_positions: np.ndarray,
        curr_positions: np.ndarray,
        actions: List[np.ndarray],
    ) -> Tuple[List[float], dict]:
        """Tum robotlar icin odul hesapla / Compute rewards for all robots."""
        dt = self.model.opt.timestep * self.frame_skip

        # Ortalama ileri hiz / Average forward velocity
        velocities = (curr_positions - prev_positions) / dt
        avg_forward_vel = np.mean(velocities[:, 0])

        # Formation hatasi / Formation error
        centroid = np.mean(curr_positions[:, :2], axis=0)
        relative_positions = curr_positions[:, :2] - centroid
        formation_error = 0.0
        for i in range(self.num_robots):
            target_rel = self._formation_offsets[i]
            error = np.sum((relative_positions[i] - target_rel) ** 2)
            formation_error += error

        # Robot bazinda odul / Per-robot reward
        rewards = []
        for i in range(self.num_robots):
            forward_reward = velocities[i, 0] * self._forward_reward_weight
            formation_reward = -formation_error * self._formation_reward_weight
            ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(actions[i]))

            # Saglik kontrolu / Health check
            z = curr_positions[i, 2]
            is_healthy = self._healthy_z_range[0] <= z <= self._healthy_z_range[1]
            healthy_reward = self._healthy_reward if is_healthy else 0.0

            reward = forward_reward + formation_reward + healthy_reward - ctrl_cost
            rewards.append(reward)

        info = {
            "avg_forward_vel": avg_forward_vel,
            "formation_error": formation_error,
        }

        return rewards, info

    def _check_termination(self) -> bool:
        """Herhangi bir robot sagliksiz mi kontrol et / Check if any robot is unhealthy."""
        for i in range(self.num_robots):
            qpos = self._get_robot_qpos(i)
            z = qpos[2]
            state = np.concatenate([self._get_robot_qpos(i), self._get_robot_qvel(i)])
            if not np.isfinite(state).all():
                return True
            if z < self._healthy_z_range[0] or z > self._healthy_z_range[1]:
                return True
        return False

    def get_robot_obs_for_agent(self, robot_idx: int) -> np.ndarray:
        """Belirli bir robot icin agent gozlemi / Get agent observation for specific robot."""
        return self._get_all_observations()[robot_idx]

    def get_global_state(self) -> np.ndarray:
        """Tum durum (centralized critic icin) / Full state for centralized critic."""
        all_obs = []
        for i in range(self.num_robots):
            all_obs.append(self._get_robot_obs(i))
        return np.concatenate(all_obs)

    def close(self):
        """Ortami kapat / Close environment."""
        pass


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Robot Quadruped Environment Test")
    print("=" * 60)

    env = MultiRobotQuadrupedEnv(num_robots=2, formation_type="line")

    print(f"\nRobot Sayisi:        {env.num_robots}")
    print(f"Formation:           {env.formation_type}")
    print(f"Robot Obs Dim:       {env.obs_dim_per_robot}")
    print(f"Action Dim/Robot:    {env.action_dim_per_robot}")
    print(f"Global State Dim:    {env.get_global_state().shape}")

    observations = env.reset(seed=42)
    for i, obs in enumerate(observations):
        print(f"\nRobot {i} Obs Shape: {obs.shape}")
        print(f"Robot {i} Obs[:5]:   {obs[:5]}")

    # Random agent testi
    print("\n" + "-" * 40)
    print("Random Agent Testi (50 adim):")
    total_rewards = [0.0] * env.num_robots
    for step in range(50):
        actions = [np.random.uniform(-1, 1, size=12) for _ in range(env.num_robots)]
        observations, rewards, done, info = env.step(actions)
        for i in range(env.num_robots):
            total_rewards[i] += rewards[i]
        if done:
            print(f"  Done at step {step + 1}")
            break

    for i in range(env.num_robots):
        print(f"  Robot {i} Toplam Reward: {total_rewards[i]:.2f}")
    print(f"  Formation Error: {info['formation_error']:.4f}")
    print(f"  Avg Forward Vel: {info['avg_forward_vel']:.4f}")

    env.close()
    print("\nTest tamamlandi!")
