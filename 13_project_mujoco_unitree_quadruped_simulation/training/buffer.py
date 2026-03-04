"""
Rollout Buffer ve GAE Hesaplama / Rollout Buffer and GAE Computation

Trajectory verilerini depolayan buffer siniflari ve
Generalized Advantage Estimation (GAE) hesaplama fonksiyonu.

Buffer classes for storing trajectory data and
Generalized Advantage Estimation (GAE) computation.
"""

import numpy as np
import torch
from typing import Generator, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# GAE Hesaplama / GAE Computation
# =============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_value: float,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE) hesapla.
    Compute Generalized Advantage Estimation.

    GAE(gamma, lambda) = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: (T,) boyutunda oduller
        values: (T,) boyutunda deger tahminleri
        next_value: Son durumun deger tahmini
        dones: (T,) boyutunda bolum bitis flag'leri
        gamma: Indirim faktoru
        lam: GAE lambda parametresi

    Returns:
        advantages: (T,) boyutunda avantaj degerleri
        returns: (T,) boyutunda hedef degerler (advantages + values)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)

    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# =============================================================================
# Tek-Agent Rollout Buffer / Single-Agent Rollout Buffer
# =============================================================================

class RolloutBuffer:
    """
    PPO icin rollout veri deposu.
    Rollout data storage for PPO.

    Trajectory boyunca obs, action, reward, value, log_prob, done depolar.
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        """Buffer'i temizle / Clear buffer."""
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Bir transition ekle / Add a transition."""
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)
        self.ptr += 1

    @property
    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        """
        GAE ile avantaj ve return hesapla.
        Compute advantages and returns using GAE.
        """
        self.advantages, self.returns = compute_gae(
            self.rewards[:self.ptr],
            self.values[:self.ptr],
            next_value,
            self.dones[:self.ptr],
            gamma,
            lam,
        )

    def get_minibatches(
        self,
        minibatch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Generator:
        """
        Rastgele minibatch'ler uret / Generate random minibatches.

        Yields:
            (obs, actions, old_log_probs, advantages, returns) tuple'lari
        """
        n = self.ptr
        indices = np.random.permutation(n)

        # Avantajlari normalize et / Normalize advantages
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_idx = indices[start:end]

            yield (
                torch.FloatTensor(self.observations[batch_idx]).to(device),
                torch.FloatTensor(self.actions[batch_idx]).to(device),
                torch.FloatTensor(self.log_probs[batch_idx]).to(device),
                torch.FloatTensor(adv[batch_idx]).to(device),
                torch.FloatTensor(self.returns[batch_idx]).to(device),
            )


# =============================================================================
# Multi-Agent Rollout Buffer
# =============================================================================

class MultiAgentRolloutBuffer:
    """
    Multi-agent senaryolar icin rollout buffer.
    Rollout buffer for multi-agent scenarios.

    Her agent icin ayri local obs ve action depolar,
    ayrica global obs (centralized critic icin) depolar.
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        local_obs_dim: int,
        global_obs_dim: int,
        action_dim_per_agent: int,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim_per_agent
        self.reset()

    def reset(self):
        """Buffer'i temizle / Clear buffer."""
        B = self.buffer_size
        N = self.num_agents

        self.local_obs = np.zeros((B, N, self.local_obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((B, self.global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((B, N, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(B, dtype=np.float32)  # Paylasilan odul / Shared reward
        self.values = np.zeros(B, dtype=np.float32)   # Centralized critic degeri
        self.log_probs = np.zeros((B, N), dtype=np.float32)
        self.dones = np.zeros(B, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        local_obs_list: list,
        global_obs: np.ndarray,
        actions_list: list,
        reward: float,
        value: float,
        log_probs_list: list,
        done: bool,
    ):
        """Bir multi-agent transition ekle / Add a multi-agent transition."""
        idx = self.ptr
        for i in range(self.num_agents):
            self.local_obs[idx, i] = local_obs_list[i]
            self.actions[idx, i] = actions_list[i]
            self.log_probs[idx, i] = log_probs_list[i]
        self.global_obs[idx] = global_obs
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.ptr += 1

    @property
    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        """GAE hesapla (centralized critic degerlerini kullanarak)."""
        self.advantages, self.returns = compute_gae(
            self.rewards[:self.ptr],
            self.values[:self.ptr],
            next_value,
            self.dones[:self.ptr],
            gamma,
            lam,
        )

    def get_minibatches(
        self,
        minibatch_size: int,
        agent_idx: int = None,
        device: torch.device = torch.device("cpu"),
    ) -> Generator:
        """
        Minibatch uret / Generate minibatches.

        Args:
            minibatch_size: Minibatch boyutu
            agent_idx: Belirli bir agent icin (None ise tumu)
            device: Torch device

        Yields:
            (local_obs, global_obs, actions, old_log_probs, advantages, returns)
        """
        n = self.ptr
        indices = np.random.permutation(n)

        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_idx = indices[start:end]

            if agent_idx is not None:
                local = self.local_obs[batch_idx, agent_idx]
                acts = self.actions[batch_idx, agent_idx]
                lps = self.log_probs[batch_idx, agent_idx]
            else:
                local = self.local_obs[batch_idx]
                acts = self.actions[batch_idx]
                lps = self.log_probs[batch_idx]

            yield (
                torch.FloatTensor(local).to(device),
                torch.FloatTensor(self.global_obs[batch_idx]).to(device),
                torch.FloatTensor(acts).to(device),
                torch.FloatTensor(lps).to(device),
                torch.FloatTensor(adv[batch_idx]).to(device),
                torch.FloatTensor(self.returns[batch_idx]).to(device),
            )
