"""
Tek-Agent PPO / Single-Agent PPO

Tum 12 eklemi tek bir policy ile kontrol eden PPO agent.
PPO agent that controls all 12 joints with a single policy.
Multi-agent yaklasimlar icin benchmark gorevi gorur.

This serves as baseline benchmark for multi-agent approaches.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from agents.networks import ContinuousActorCritic


class PPOAgent:
    """
    PPO-Clip ile continuous control agent.
    Continuous control agent with PPO-Clip.
    """

    def __init__(
        self,
        obs_dim: int = 39,
        action_dim: int = 12,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.0,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        self.network = ContinuousActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Gozlemden aksiyon sec.
        Select action from observation.

        Returns:
            action (np.ndarray), log_prob (float), value (float)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_t, deterministic=deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        """
        PPO-Clip guncelleme adimi.
        PPO-Clip update step.
        """
        _, new_log_probs, entropy, values = self.network.get_action_and_value(obs, actions)

        # Policy loss (clipped surrogate)
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Toplam kayip / Total loss
        loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def get_state_dict(self) -> dict:
        """Model state dict'i dondur / Return model state dict."""
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        """Model state dict'i yukle / Load model state dict."""
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
