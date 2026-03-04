"""
Robot-Arasi Iletisim Modulu / Inter-Robot Communication Module

Coklu robot senaryosunda agentlarin birbirleriyle
ogrenilmis mesajlar ile iletisim kurmasini saglar.

Enables learned message passing between agents
in multi-robot scenarios.

Protokol / Protocol:
    1. Her agent gozlemini encode eder -> hidden state
    2. Hidden state'den mesaj uretilir
    3. Mesajlar agentlar arasi paylasılir
    4. Alinan mesajlar + hidden state -> aksiyon
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from agents.networks import CommActorCritic


class CommMAPPOAgent:
    """
    Iletisim destekli MAPPO agent (coklu robot kontrolu).
    Communication-enabled MAPPO agent for multi-robot control.

    Her robot bir CommMAPPOAgent instance'i ile kontrol edilir.
    Agentlar mesaj alisverisi yaparak koordine olur.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 12,
        global_obs_dim: int = 78,
        message_dim: int = 16,
        hidden_dim: int = 256,
        num_agents: int = 2,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")
        self.hidden_dim = hidden_dim

        self.network = CommActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_obs_dim=global_obs_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """Gozlemi encode et / Encode observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.network.encode(obs_t)
        return hidden.squeeze(0)  # (hidden_dim,)

    def produce_message(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mesaj uret / Produce message from hidden state."""
        with torch.no_grad():
            msg = self.network.comm.encode_message(hidden.unsqueeze(0))
        return msg.squeeze(0)

    def act_with_comm(
        self,
        hidden: torch.Tensor,
        other_messages: List[torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Iletisim sonrasi aksiyon sec.
        Select action after receiving messages.
        """
        with torch.no_grad():
            # Mesajlari coz / Decode messages
            msgs_unsqueezed = [m.unsqueeze(0) for m in other_messages]
            comm_signal = self.network.comm.decode_messages(msgs_unsqueezed)

            # Aksiyon sec / Select action
            action, log_prob, _ = self.network.act(
                hidden.unsqueeze(0), comm_signal, deterministic=deterministic
            )

        return action.squeeze(0).cpu().numpy(), log_prob.item()

    def get_value(self, global_obs: np.ndarray) -> float:
        """Centralized critic degeri / Centralized critic value."""
        obs_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.network.get_value(obs_t)
        return value.item()

    def ppo_update(
        self,
        obs_arr: np.ndarray,
        actions_arr: np.ndarray,
        old_log_probs_arr: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        global_obs_arr: np.ndarray,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
    ):
        """
        PPO-Clip guncelleme (basitlestirilmis: iletisim olmadan).
        PPO-Clip update (simplified: without communication during update).

        Egitim sirasinda iletisim rollout'ta yapilir,
        guncelleme basitlik icin sadece obs->action mapper olarak yapilir.
        """
        n = len(obs_arr)
        obs_t = torch.FloatTensor(obs_arr).to(self.device)
        actions_t = torch.FloatTensor(actions_arr).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs_arr).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        global_t = torch.FloatTensor(global_obs_arr).to(self.device)

        for _ in range(ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                end = min(start + minibatch_size, n)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_actions = actions_t[idx]
                b_old_lp = old_lp_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]
                b_global = global_t[idx]

                # Forward pass (iletisim olmadan, basitlestirilmis)
                hidden = self.network.encode(b_obs)
                zero_comm = torch.zeros(len(idx), self.hidden_dim).to(self.device)
                _, new_lp, entropy = self.network.act(hidden, zero_comm, b_actions)

                # Policy loss
                ratio = (new_lp - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.network.get_value(b_global)
                value_loss = nn.functional.mse_loss(values, b_ret)

                # Entropy
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def get_state_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
