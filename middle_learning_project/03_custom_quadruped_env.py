#!/usr/bin/env python3
"""
Ozel Dort Bacakli Robot Ortami / Custom Quadruped Robot Environment

Bu modul, 02_custom_quadruped_model.xml dosyasini yukleyen ve ileri yurume
gorevi icin ozel bir Gymnasium ortami tanimlar. Ant-v5'e benzer bir yapiyla
calisan bu ortam, kopek benzeri dikdortgen govdeli bir dort bacakli robotu
kontrol eder.

This module defines a custom Gymnasium environment that loads the
02_custom_quadruped_model.xml file and implements a forward locomotion task.
Similar in structure to Ant-v5, this environment controls a dog-like
quadruped robot with a rectangular body.

Observation Space (27 eleman / elements):
    - z-pozisyon (1): Govde yuksekligi / Torso height
    - quaternion (4): Govde yonelimi / Torso orientation
    - eklem acilari (8): 4 bacak x 2 eklem / 4 legs x 2 joints
    - govde hizlari (6): Lineer + acisal hiz / Linear + angular velocity
    - eklem hizlari (8): 8 eklem hizi / 8 joint velocities

Action Space:
    - Box(-1, 1, (8,)): 8 motor torku / 8 motor torques
      [fl_hip, fl_knee, fr_hip, fr_knee, bl_hip, bl_knee, br_hip, br_knee]

Reward:
    reward = forward_reward + healthy_reward - ctrl_cost
    - forward_reward: x yonundeki hiz * agirlik / x-velocity * weight
    - healthy_reward: Robot ayakta ise sabit odul / Fixed reward if healthy
    - ctrl_cost: Aksiyon buyuklugu cezasi / Action magnitude penalty
"""

from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CustomQuadrupedEnv(MujocoEnv, utils.EzPickle):
    """
    Ozel dort bacakli robot ortami / Custom quadruped robot environment.

    Args:
        xml_file: MuJoCo XML model dosyasi yolu / Path to MuJoCo XML model file
        frame_skip: Simulasyon adim atlama sayisi / Number of simulation steps per action
        forward_reward_weight: Ileri odul agirligi / Forward reward weight
        ctrl_cost_weight: Kontrol cezasi agirligi / Control cost weight
        healthy_reward: Saglik odulu degeri / Healthy reward value
        terminate_when_unhealthy: Sagliksiz oldugunda bitir / Terminate when unhealthy
        healthy_z_range: Saglikli z-koordinat araligi / Healthy z-coordinate range
        reset_noise_scale: Reset gurultu olcegi / Reset noise scale
        exclude_current_positions_from_observation: x,y pozisyonlarini cikar /
            Exclude x,y positions from observation
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.15, 0.6),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # XML dosya yolunu belirle / Determine XML file path
        if xml_file is None:
            xml_file = str(Path(__file__).parent / "02_custom_quadruped_model.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # asagida tanimlanacak / defined below
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Gozlem uzayini hesapla / Calculate observation space
        # qpos: 7 (free joint: xyz + quaternion) + 8 (hinge joints) = 15
        # qvel: 6 (free joint: linear + angular) + 8 (hinge joints) = 14
        obs_size = self.data.qpos.size + self.data.qvel.size
        if exclude_current_positions_from_observation:
            obs_size -= 2  # x, y koordinatlarini cikar / exclude x, y

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    @property
    def is_healthy(self):
        """Robot saglikli mi kontrol et / Check if robot is healthy"""
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def healthy_reward(self):
        """Saglik odulunu hesapla / Calculate healthy reward"""
        return self._healthy_reward if self.is_healthy else 0.0

    def control_cost(self, action):
        """Kontrol cezasini hesapla / Calculate control cost"""
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_rew(self, x_velocity: float, action):
        """
        Odul fonksiyonu / Reward function

        reward = forward_reward + healthy_reward - ctrl_cost

        Args:
            x_velocity: x yonundeki hiz / Velocity in x direction
            action: Uygulanan aksiyon / Applied action

        Returns:
            reward: Toplam odul / Total reward
            reward_info: Odul bileşenleri / Reward components
        """
        # 1. Ileri odul / Forward reward
        forward_reward = x_velocity * self._forward_reward_weight

        # 2. Saglik odulu / Healthy reward
        healthy_reward = self.healthy_reward

        # 3. Kontrol cezasi / Control cost
        ctrl_cost = self.control_cost(action)

        reward = forward_reward + healthy_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def step(self, action):
        """
        Bir simulasyon adimi at / Take one simulation step

        Args:
            action: 8 motorun torklari [-1, 1] / 8 motor torques [-1, 1]

        Returns:
            observation: Gozlem vektoru / Observation vector
            reward: Odul degeri / Reward value
            terminated: Bolum bitti mi / Episode terminated
            truncated: Her zaman False (TimeLimit wrapper halleder) / Always False
            info: Ek bilgiler / Additional info
        """
        # Onceki x-y pozisyonu kaydet / Record x-y position before
        xy_position_before = self.data.body("torso").xpos[:2].copy()

        # Simulasyonu calistir / Run simulation
        self.do_simulation(action, self.frame_skip)

        # Sonraki x-y pozisyonu / x-y position after
        xy_position_after = self.data.body("torso").xpos[:2].copy()

        # Hiz hesapla / Calculate velocity
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Gozlem ve odul / Observation and reward
        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)

        # Bitirme kontrolu / Termination check
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        """
        Gozlem vektorunu olustur / Construct observation vector

        Gozlem icerigi / Observation content:
            - qpos[2:]: z-pozisyon, quaternion, eklem acilari (x,y haric)
            - qvel: Govde ve eklem hizlari
        """
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]  # x, y koordinatlarini cikar / skip x, y

        return np.concatenate((position, velocity))

    def reset_model(self):
        """
        Modeli sifirla / Reset the model

        Baslangic pozisyonuna gurultu ekleyerek sifirla.
        Reset to initial position with added noise.
        """
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Pozisyon: uniform noise ekle / Position: add uniform noise
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        # Hiz: Gaussian noise ekle / Velocity: add Gaussian noise
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        """Reset sonrasi bilgi / Info after reset"""
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
