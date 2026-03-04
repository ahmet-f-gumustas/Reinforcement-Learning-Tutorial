"""
Multi-Agent PPO (MAPPO) - Per-Leg CTDE / Multi-Agent PPO with Per-Leg CTDE

4 agent, her biri bir bacagi kontrol eder (3 eklem).
Centralized Training, Decentralized Execution (CTDE) mimarisi.

4 agents, each controlling one leg (3 joints).
Centralized Training, Decentralized Execution (CTDE) architecture.

Mimari / Architecture:
    - 4x LegActor: Lokal gozlemden aksiyon (decentralized)
    - 1x CentralizedCritic: Global gozlemden deger (centralized)
    - Tum agentlar ayni odul'u paylasilir (cooperative)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from agents.networks import LegActor, CentralizedCritic


class MAPPOLegAgent:
    """
    Tek bacak icin MAPPO agent.
    MAPPO agent for a single leg.

    Kendi LegActor'u var, paylasilmis CentralizedCritic'e referans tutar.
    """

    def __init__(
        self,
        agent_id: int,
        local_obs_dim: int = 18,
        action_dim: int = 3,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        device: torch.device = None,
    ):
        self.agent_id = agent_id
        self.device = device or torch.device("cpu")

        self.actor = LegActor(local_obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)

    def select_action(
        self,
        local_obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Lokal gozlemden aksiyon sec / Select action from local observation."""
        obs_t = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.actor(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), log_prob.item()


class MAPPOController:
    """
    4 bacak agent'ini yoneten MAPPO kontrolcusu.
    MAPPO controller managing 4 leg agents.

    Centralized critic paylasimli, her actor bagimsiz guncellenir.
    Shared centralized critic, each actor updated independently.
    """

    def __init__(
        self,
        num_agents: int = 4,
        local_obs_dim: int = 18,
        global_obs_dim: int = 39,
        action_dim: int = 3,
        actor_hidden: int = 128,
        critic_hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        self.num_agents = num_agents
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        # 4 ayri actor / 4 separate actors
        self.agents = [
            MAPPOLegAgent(
                agent_id=i,
                local_obs_dim=local_obs_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden,
                lr=lr_actor,
                device=self.device,
            )
            for i in range(num_agents)
        ]

        # 1 paylasimli centralized critic / 1 shared centralized critic
        self.critic = CentralizedCritic(global_obs_dim, critic_hidden).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=1e-5
        )

    def select_actions(
        self,
        local_obs_list: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Tum agentlar icin aksiyon sec.
        Select actions for all agents.

        Args:
            local_obs_list: Her agent icin lokal gozlem listesi (4 x 18-dim)

        Returns:
            actions_list: Her agent'in aksiyonu (4 x 3-dim)
            log_probs_list: Her agent'in log probability'si
        """
        actions = []
        log_probs = []
        for i in range(self.num_agents):
            action, lp = self.agents[i].select_action(
                local_obs_list[i], deterministic=deterministic
            )
            actions.append(action)
            log_probs.append(lp)
        return actions, log_probs

    def get_value(self, global_obs: np.ndarray) -> float:
        """Centralized critic ile deger tahmini / Value prediction with centralized critic."""
        obs_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(obs_t)
        return value.item()

    def update_critic(self, global_obs: torch.Tensor, returns: torch.Tensor):
        """
        Centralized critic'i guncelle.
        Update centralized critic.
        """
        values = self.critic(global_obs)
        critic_loss = nn.functional.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def update_actor(
        self,
        agent_idx: int,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """
        Belirli bir agent'in actor'unu guncelle (PPO-Clip).
        Update a specific agent's actor using PPO-Clip.
        """
        agent = self.agents[agent_idx]

        _, new_log_probs, entropy = agent.actor(local_obs, actions)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + self.entropy_coeff * entropy_loss

        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
        agent.optimizer.step()

    def get_state_dict(self) -> dict:
        """Tum model state dict'lerini dondur / Return all model state dicts."""
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            **{
                f"actor_{i}": self.agents[i].actor.state_dict()
                for i in range(self.num_agents)
            },
            **{
                f"actor_optimizer_{i}": self.agents[i].optimizer.state_dict()
                for i in range(self.num_agents)
            },
        }

    def load_state_dict(self, state_dict: dict):
        """Tum model state dict'lerini yukle / Load all model state dicts."""
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        for i in range(self.num_agents):
            self.agents[i].actor.load_state_dict(state_dict[f"actor_{i}"])
            self.agents[i].optimizer.load_state_dict(state_dict[f"actor_optimizer_{i}"])
