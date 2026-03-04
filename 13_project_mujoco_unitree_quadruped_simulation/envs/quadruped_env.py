"""
Unitree Go2 Ilhamli Quadruped Ortami / Unitree Go2 Inspired Quadruped Environment

Bu modul, Unitree Go2 tarzinda 12 DOF dort bacakli robot icin
ozel bir Gymnasium ortami tanimlar. Hem tek-agent hem de multi-agent
kullanimini destekler.

This module defines a custom Gymnasium environment for a Unitree Go2-style
12 DOF quadruped robot. Supports both single-agent and multi-agent usage.

Observation Space (39 eleman / elements):
    - z-pozisyon (1): Govde yuksekligi / Torso height
    - quaternion (4): Govde yonelimi / Torso orientation
    - eklem acilari (12): 4 bacak x 3 eklem / 4 legs x 3 joints
    - govde hizlari (6): Lineer + acisal hiz / Linear + angular velocity
    - eklem hizlari (12): 12 eklem hizi / 12 joint velocities
    - ayak temasi (4): Her ayagin temas durumu / Foot contact states

Action Space:
    - Box(-1, 1, (12,)): 12 motor torku / 12 motor torques
      [fl_abd, fl_hip, fl_knee, fr_abd, fr_hip, fr_knee,
       bl_abd, bl_hip, bl_knee, br_abd, br_hip, br_knee]

Reward:
    reward = forward_reward + healthy_reward - ctrl_cost - energy_cost
"""

import numpy as np
from typing import Dict, Tuple, Union
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# Bacak isimleri ve eklem indeksleri / Leg names and joint indices
# qpos: [root(7), fl_abd, fl_hip, fl_knee, fr_abd, fr_hip, fr_knee,
#         bl_abd, bl_hip, bl_knee, br_abd, br_hip, br_knee] = 19 total
# qvel: [root(6), fl_abd, fl_hip, fl_knee, fr_abd, fr_hip, fr_knee,
#         bl_abd, bl_hip, bl_knee, br_abd, br_hip, br_knee] = 18 total
LEG_NAMES = ["FL", "FR", "BL", "BR"]
LEG_JOINT_INDICES = {
    "FL": [0, 1, 2],    # fl_abd, fl_hip, fl_knee (qpos icinde 7+offset)
    "FR": [3, 4, 5],
    "BL": [6, 7, 8],
    "BR": [9, 10, 11],
}


class UnitreeQuadrupedEnv(MujocoEnv, utils.EzPickle):
    """
    Unitree Go2 ilhamli dort bacakli robot ortami.
    Unitree Go2 inspired quadruped robot environment.

    Hem tek-agent (tum 12 eklemi kontrol) hem de multi-agent
    (bacak bazinda kontrol) kullanimini destekler.

    Supports both single-agent (controls all 12 joints) and
    multi-agent (per-leg control) usage patterns.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 10,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.05,
        healthy_reward: float = 1.0,
        energy_cost_weight: float = 0.01,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.15, 0.55),
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, default_camera_config,
            forward_reward_weight, ctrl_cost_weight, healthy_reward,
            energy_cost_weight, terminate_when_unhealthy,
            healthy_z_range, reset_noise_scale, **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._energy_cost_weight = energy_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        if xml_file is None:
            xml_file = str(Path(__file__).parent.parent / "models" / "unitree_go2.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Gozlem uzayi: z(1) + quat(4) + joints(12) + body_vel(6) + joint_vel(12) + contacts(4) = 39
        obs_size = 39
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Sensor indekslerini kaydet / Cache sensor indices
        self._foot_sensor_names = [
            "fl_foot_touch", "fr_foot_touch", "bl_foot_touch", "br_foot_touch"
        ]

    # =========================================================================
    # Temel Ortam Metotlari / Core Environment Methods
    # =========================================================================

    @property
    def is_healthy(self) -> bool:
        """Robot saglikli mi kontrol et / Check if robot is healthy."""
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def healthy_reward(self) -> float:
        return self._healthy_reward if self.is_healthy else 0.0

    def control_cost(self, action) -> float:
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def energy_cost(self, action) -> float:
        """Enerji maliyeti: |torque * joint_velocity| / Energy cost."""
        joint_vel = self.data.qvel[6:]  # 12 eklem hizi
        return self._energy_cost_weight * np.sum(np.abs(action * joint_vel))

    def step(self, action):
        """Bir simulasyon adimi at / Take one simulation step."""
        xy_position_before = self.data.body("torso").xpos[:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_position_after = self.data.body("torso").xpos[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity = xy_velocity[0]

        observation = self._get_obs()

        # Odul hesaplama / Reward calculation
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        energy_cost = self.energy_cost(action)

        reward = forward_reward + healthy_reward - ctrl_cost - energy_cost

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "x_velocity": x_velocity,
            "y_velocity": xy_velocity[1],
            "reward_forward": forward_reward,
            "reward_healthy": healthy_reward,
            "cost_ctrl": -ctrl_cost,
            "cost_energy": -energy_cost,
            "foot_contacts": self._get_foot_contacts(),
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """
        Gozlem vektoru olustur / Construct observation vector (39 dim).

        [z_pos(1), quat(4), joint_angles(12), body_vel(6), joint_vel(12), contacts(4)]
        """
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        z_pos = qpos[2:3]                  # 1
        orientation = qpos[3:7]             # 4 (quaternion)
        joint_angles = qpos[7:]             # 12
        body_vel = qvel[:6]                 # 6 (linear + angular)
        joint_vel = qvel[6:]                # 12
        foot_contacts = self._get_foot_contacts()  # 4

        return np.concatenate([
            z_pos, orientation, joint_angles,
            body_vel, joint_vel, foot_contacts
        ])

    def _get_foot_contacts(self) -> np.ndarray:
        """Ayak temas durumlarini oku / Read foot contact states."""
        contacts = np.zeros(4, dtype=np.float64)
        for i, name in enumerate(self._foot_sensor_names):
            sensor_id = self.model.sensor(name).id
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            contact_force = self.data.sensordata[sensor_adr:sensor_adr + sensor_dim]
            contacts[i] = 1.0 if np.sum(np.abs(contact_force)) > 0.1 else 0.0
        return contacts

    def reset_model(self):
        """Modeli sifirla / Reset the model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    # =========================================================================
    # Multi-Agent Yardimci Metotlar / Multi-Agent Helper Methods
    # =========================================================================

    def get_leg_obs(self, leg_idx: int) -> np.ndarray:
        """
        Belirli bir bacak icin lokal gozlem cikar.
        Extract local observation for a specific leg.

        Args:
            leg_idx: 0=FL, 1=FR, 2=BL, 3=BR

        Returns:
            18 boyutlu vektor / 18-dim vector:
            [z(1), quat(4), body_vel(6)] + [leg_joints(3), leg_vels(3), contact(1)]
            = shared(11) + local(7) = 18
        """
        full_obs = self._get_obs()

        # Paylasilan govde durumu / Shared torso state (11 dim)
        z_pos = full_obs[0:1]           # 1
        orientation = full_obs[1:5]     # 4
        body_vel = full_obs[17:23]      # 6
        shared_state = np.concatenate([z_pos, orientation, body_vel])  # 11

        # Bu bacagin eklem acilari / This leg's joint angles (3 dim)
        joint_start = 5 + leg_idx * 3
        leg_joints = full_obs[joint_start:joint_start + 3]

        # Bu bacagin eklem hizlari / This leg's joint velocities (3 dim)
        vel_start = 23 + leg_idx * 3
        leg_vels = full_obs[vel_start:vel_start + 3]

        # Bu bacagin temas durumu / This leg's foot contact (1 dim)
        contact = full_obs[35 + leg_idx:35 + leg_idx + 1]

        return np.concatenate([shared_state, leg_joints, leg_vels, contact])  # 18

    def get_global_state(self) -> np.ndarray:
        """
        Tam gozlem vektoru (centralized critic icin).
        Full observation vector (for centralized critic).

        Returns:
            39 boyutlu gozlem / 39-dim observation
        """
        return self._get_obs()

    def get_torso_state(self) -> np.ndarray:
        """
        Govde durumu (hiyerarsik manager icin).
        Torso state (for hierarchical manager).

        Returns:
            11 boyutlu vektor / 11-dim vector:
            [z(1), quat(4), body_vel(6)]
        """
        full_obs = self._get_obs()
        z_pos = full_obs[0:1]
        orientation = full_obs[1:5]
        body_vel = full_obs[17:23]
        return np.concatenate([z_pos, orientation, body_vel])


# =============================================================================
# Test / Standalone Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Unitree Go2 Quadruped Environment Test")
    print("=" * 60)

    env = UnitreeQuadrupedEnv()

    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Action Shape:      {env.action_space.shape}")

    obs, info = env.reset()
    print(f"\nIlk Gozlem (ilk 10): {obs[:10]}")
    print(f"Gozlem Boyutu:       {obs.shape}")

    # Multi-agent gozlemler
    for i, name in enumerate(LEG_NAMES):
        leg_obs = env.get_leg_obs(i)
        print(f"\n{name} Bacak Gozlemi ({leg_obs.shape}): {leg_obs[:5]}...")

    print(f"\nGlobal State Shape:  {env.get_global_state().shape}")
    print(f"Torso State Shape:   {env.get_torso_state().shape}")

    # Random agent testi
    print("\n" + "-" * 40)
    print("Random Agent Testi (100 adim):")
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Terminated at step {step + 1}")
            obs, info = env.reset()
            break

    print(f"  Toplam Reward: {total_reward:.2f}")
    print(f"  Son x_position: {info.get('x_position', 0):.4f}")
    print(f"  Son x_velocity: {info.get('x_velocity', 0):.4f}")

    env.close()
    print("\nTest tamamlandi!")
