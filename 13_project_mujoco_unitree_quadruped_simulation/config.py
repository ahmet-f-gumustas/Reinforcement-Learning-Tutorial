"""
Proje Konfigurasyonu / Project Configuration

Tum hyperparameter ve ayarlar tek bir dosyada tanimlanir.
All hyperparameters and settings are defined in a single file.
"""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
SAVE_DIR = PROJECT_DIR / "saved_models"


# =============================================================================
# Ortam Ayarlari / Environment Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """MuJoCo quadruped ortam ayarlari."""
    xml_path: str = str(MODELS_DIR / "unitree_go2.xml")
    frame_skip: int = 10
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.05
    healthy_reward: float = 1.0
    energy_cost_weight: float = 0.01
    terminate_when_unhealthy: bool = True
    healthy_z_range: Tuple[float, float] = (0.15, 0.55)
    reset_noise_scale: float = 0.1
    max_episode_steps: int = 1000


# =============================================================================
# PPO Hyperparametreleri / PPO Hyperparameters
# =============================================================================

@dataclass
class PPOConfig:
    """Tek-agent PPO icin hyperparametreler."""
    # Ag mimarisi / Network
    hidden_dim: int = 256
    num_layers: int = 2

    # Ogrenme / Learning
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.0
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Rollout
    rollout_length: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 64

    # Egitim / Training
    total_timesteps: int = 500_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    seed: int = 42


# =============================================================================
# MAPPO Hyperparametreleri / MAPPO Hyperparameters
# =============================================================================

@dataclass
class MAPPOConfig:
    """Multi-Agent PPO (per-leg CTDE) icin hyperparametreler."""
    num_agents: int = 4  # 4 bacak / 4 legs
    local_obs_dim: int = 18  # torso(11) + leg_joints(3) + leg_vels(3) + contact(1)
    global_obs_dim: int = 39  # full observation
    action_dim_per_agent: int = 3  # 3 eklem per bacak / 3 joints per leg

    # Ag mimarisi / Network
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256

    # PPO ayarlari
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.15          # 0.2 -> 0.15: daha kucuk adimlar, collapse riski azalir
    entropy_coeff: float = 0.02     # 0.01 -> 0.02: exploration'i korur, collapse onler
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Rollout
    rollout_length: int = 2048
    ppo_epochs: int = 8             # 10 -> 8: ayni veriyle daha az tekrar, overfitting azalir
    minibatch_size: int = 128       # 64 -> 128: daha stabil gradyan tahmini

    # Egitim
    total_timesteps: int = 500_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    seed: int = 42


# =============================================================================
# Hiyerarsik Kontrol / Hierarchical Control Config
# =============================================================================

@dataclass
class HierarchicalConfig:
    """Manager-Worker hiyerarsik kontrol icin hyperparametreler."""
    # Manager (gait planner)
    manager_obs_dim: int = 15   # torso(11) + avg_vel(2) + phase(2)
    manager_action_dim: int = 6  # 4 phase_offsets + frequency + step_height
    manager_hidden_dim: int = 128
    manager_decision_period: int = 50  # K adimda bir karar
    lr_manager: float = 3e-4

    # Worker (leg controllers)
    worker_obs_dim: int = 26    # local_obs(18) + gait_embed(8)
    worker_action_dim: int = 3
    worker_hidden_dim: int = 128
    gait_embed_dim: int = 8
    lr_worker: float = 3e-4

    # Ortak PPO ayarlari
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # Rollout
    rollout_length: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 64

    # Egitim
    total_timesteps: int = 500_000
    pretrain_worker_steps: int = 50_000  # Sabit gait ile on-egitim
    eval_interval: int = 10_000
    save_interval: int = 50_000
    seed: int = 42


# =============================================================================
# Coklu Robot / Multi-Robot Config
# =============================================================================

@dataclass
class MultiRobotConfig:
    """Coklu robot formation kontrolu icin hyperparametreler."""
    num_robots: int = 2
    formation_type: str = "line"  # "line", "triangle"
    formation_spacing: float = 1.5  # metre

    # Gozlem / Observation
    robot_obs_dim: int = 39
    relative_obs_dim: int = 6  # diger robotun x,y,z pos + vel
    message_dim: int = 16

    # Ag mimarisi
    hidden_dim: int = 256
    comm_rounds: int = 1

    # Odul agirliklari / Reward weights
    forward_weight: float = 1.0
    formation_weight: float = 2.0
    comm_cost_weight: float = 0.001

    # PPO ayarlari
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # Rollout
    rollout_length: int = 1024
    ppo_epochs: int = 10
    minibatch_size: int = 64

    # Egitim
    total_timesteps: int = 300_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    seed: int = 42


# =============================================================================
# Config Factory
# =============================================================================

def get_config(agent_type: str):
    """Agent turune gore config dondur / Return config for agent type."""
    configs = {
        "single_ppo": (EnvConfig(), PPOConfig()),
        "mappo": (EnvConfig(), MAPPOConfig()),
        "hierarchical": (EnvConfig(), HierarchicalConfig()),
        "multi_robot": (EnvConfig(), MultiRobotConfig()),
    }
    if agent_type not in configs:
        raise ValueError(
            f"Bilinmeyen agent turu: {agent_type}. "
            f"Gecerli turler: {list(configs.keys())}"
        )
    return configs[agent_type]
