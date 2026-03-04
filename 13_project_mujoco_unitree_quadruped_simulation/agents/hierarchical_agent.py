"""
Hiyerarsik Multi-Agent Kontrol / Hierarchical Multi-Agent Control

Iki seviyeli hiyerarsik mimari:
  Manager (Gait Planner): Her K adimda gait parametreleri secer
  Workers (Leg Controllers): Her adimda eklem torklari uretir

Two-level hierarchical architecture:
  Manager (Gait Planner): Selects gait parameters every K steps
  Workers (Leg Controllers): Produce joint torques every step
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from agents.networks import (
    GaitPlannerNetwork,
    GaitCommandEncoder,
    HierarchicalLegActor,
    CentralizedCritic,
)


class HierarchicalController:
    """
    Manager-Worker hiyerarsik kontrolcu.
    Manager-Worker hierarchical controller.

    Manager: Govde durumundan gait komutu uretir (dusuk frekans)
    Worker: Gait komutu + lokal gozlemden eklem torku uretir (yuksek frekans)
    """

    def __init__(
        self,
        manager_obs_dim: int = 15,
        manager_action_dim: int = 6,
        manager_hidden: int = 128,
        worker_obs_dim: int = 26,
        worker_action_dim: int = 3,
        worker_hidden: int = 128,
        gait_embed_dim: int = 8,
        lr_manager: float = 3e-4,
        lr_worker: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        decision_period: int = 50,
        device: torch.device = None,
    ):
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.decision_period = decision_period
        self.device = device or torch.device("cpu")

        # Manager: Gait planner
        self.manager = GaitPlannerNetwork(
            manager_obs_dim, manager_action_dim, manager_hidden
        ).to(self.device)
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(), lr=lr_manager, eps=1e-5
        )

        # Gait command encoder
        self.gait_encoder = GaitCommandEncoder(
            manager_action_dim, gait_embed_dim
        ).to(self.device)
        self.gait_encoder_optimizer = torch.optim.Adam(
            self.gait_encoder.parameters(), lr=lr_worker, eps=1e-5
        )

        # 4 Worker: Hiyerarsik bacak kontrolculeri
        self.workers = nn.ModuleList([
            HierarchicalLegActor(
                local_obs_dim=18,  # leg_obs kendi basina
                gait_embed_dim=gait_embed_dim,
                action_dim=worker_action_dim,
                hidden_dim=worker_hidden,
            )
            for _ in range(4)
        ]).to(self.device)
        self.worker_optimizers = [
            torch.optim.Adam(w.parameters(), lr=lr_worker, eps=1e-5)
            for w in self.workers
        ]

        # Worker icin centralized critic
        self.worker_critic = CentralizedCritic(39, 256).to(self.device)
        self.worker_critic_optimizer = torch.optim.Adam(
            self.worker_critic.parameters(), lr=lr_worker, eps=1e-5
        )

    def manager_action(
        self,
        manager_obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Manager gait komutu sec / Manager selects gait command."""
        obs_t = torch.FloatTensor(manager_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.manager(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def encode_gait_command(self, gait_cmd: np.ndarray) -> np.ndarray:
        """Gait komutunu embedding'e donustur / Encode gait command to embedding."""
        cmd_t = torch.FloatTensor(gait_cmd).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed = self.gait_encoder(cmd_t)
        return embed.squeeze(0).cpu().numpy()

    def worker_actions(
        self,
        worker_obs_list: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Worker'lardan aksiyon al / Get actions from workers."""
        actions = []
        log_probs = []
        for i in range(4):
            obs_t = torch.FloatTensor(worker_obs_list[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, lp, _ = self.workers[i](obs_t, deterministic=deterministic)
            actions.append(action.squeeze(0).cpu().numpy())
            log_probs.append(lp.item())
        return actions, log_probs

    def get_worker_value(self, global_obs: np.ndarray) -> float:
        """Worker centralized critic degeri / Worker centralized critic value."""
        obs_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.worker_critic(obs_t)
        return value.item()

    def update_workers(self, batch):
        """Worker'lari PPO ile guncelle / Update workers with PPO."""
        local_obs, global_obs, actions, old_log_probs, advantages, returns = batch

        # Worker critic guncelle
        values = self.worker_critic(global_obs)
        critic_loss = nn.functional.mse_loss(values, returns)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.worker_critic.parameters(), self.max_grad_norm)
        self.worker_critic_optimizer.step()

        # Her worker'in actor'unu guncelle
        # local_obs: (batch, num_agents, obs_dim) veya (batch, obs_dim) if agent_idx specified
        if local_obs.dim() == 3:
            # Tum agentlar birden
            for i in range(4):
                agent_obs = local_obs[:, i]
                agent_actions = actions[:, i]
                agent_old_lp = old_log_probs[:, i] if old_log_probs.dim() > 1 else old_log_probs

                _, new_lp, entropy = self.workers[i](agent_obs, agent_actions)
                ratio = (new_lp - agent_old_lp).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

                self.worker_optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.workers[i].parameters(), self.max_grad_norm)
                self.worker_optimizers[i].step()

    def update_manager(
        self,
        obs_arr: np.ndarray,
        actions_arr: np.ndarray,
        log_probs_arr: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: torch.device,
    ):
        """Manager'i PPO ile guncelle / Update manager with PPO."""
        obs_t = torch.FloatTensor(obs_arr).to(device)
        actions_t = torch.FloatTensor(actions_arr).to(device)
        old_lp_t = torch.FloatTensor(log_probs_arr).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        _, new_lp, entropy, values = self.manager(obs_t, actions_t)

        ratio = (new_lp - old_lp_t).exp()
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values, ret_t)
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss

        self.manager_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
        self.manager_optimizer.step()

    def get_state_dict(self) -> dict:
        return {
            "manager": self.manager.state_dict(),
            "gait_encoder": self.gait_encoder.state_dict(),
            "workers": self.workers.state_dict(),
            "worker_critic": self.worker_critic.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.manager.load_state_dict(state_dict["manager"])
        self.gait_encoder.load_state_dict(state_dict["gait_encoder"])
        self.workers.load_state_dict(state_dict["workers"])
        self.worker_critic.load_state_dict(state_dict["worker_critic"])
