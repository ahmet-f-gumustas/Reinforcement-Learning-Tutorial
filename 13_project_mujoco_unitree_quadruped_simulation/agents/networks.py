"""
Sinir Agi Mimarileri / Neural Network Architectures

Tum agent turleri icin kullanilan sinir agi modulleri.
All neural network modules used across different agent types.

Mimariler / Architectures:
- ContinuousActorCritic: Tek-agent PPO icin
- LegActor: Per-leg decentralized actor (MAPPO)
- CentralizedCritic: CTDE shared critic
- GaitPlannerNetwork: Hiyerarsik manager
- GaitCommandEncoder: Gait komutu embedding
- HierarchicalLegActor: Gait-conditioned worker
- CommunicationModule: Robot-arasi mesajlasma
- CommActorCritic: Iletisimli multi-robot agent
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List, Optional


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Ortogonal agirlik baslatma / Orthogonal weight initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Tek-Agent PPO Aglari / Single-Agent PPO Networks
# =============================================================================

class ContinuousActorCritic(nn.Module):
    """
    Continuous action space icin Actor-Critic agi.
    Actor-Critic network for continuous action spaces.

    Actor: obs -> hidden -> mean (tanh), log_std (parameter)
    Critic: obs -> hidden -> value (ayri ag / separate network)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Actor agi / Actor network
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic agi / Critic network (ayri / separate)
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aksiyon, log_prob, entropy ve deger dondur.
        Return action, log_prob, entropy, and value.
        """
        hidden = self.actor_net(obs)
        mean = self.actor_mean(hidden)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_net(obs).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Sadece deger tahmini / Value prediction only."""
        return self.critic_net(obs).squeeze(-1)


# =============================================================================
# MAPPO Aglari / MAPPO Networks
# =============================================================================

class LegActor(nn.Module):
    """
    Tek bacak icin decentralized actor.
    Decentralized actor for a single leg.

    Girdi: Lokal gozlem (torso + bacak durumu) -> 3 eklem torku
    Input: Local observation (torso + leg state) -> 3 joint torques
    """

    def __init__(self, local_obs_dim: int = 18, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(local_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        local_obs: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log_prob, entropy."""
        hidden = self.net(local_obs)
        mean = self.mean_head(hidden)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    CTDE icin centralized critic.
    Centralized critic for CTDE that sees global state.
    """

    def __init__(self, global_obs_dim: int = 39, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# =============================================================================
# Hiyerarsik Agler / Hierarchical Networks
# =============================================================================

class GaitPlannerNetwork(nn.Module):
    """
    Ust-duzey gait planlayici (manager).
    High-level gait planner (manager).

    Girdi: Govde durumu -> gait parametreleri (phase offsets, freq, height)
    """

    def __init__(self, obs_dim: int = 15, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.value_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log_prob, entropy, value."""
        hidden = self.net(obs)
        mean = self.mean_head(hidden)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_net(obs).squeeze(-1)
        return action, log_prob, entropy, value


class GaitCommandEncoder(nn.Module):
    """
    Gait komutunu dusuk boyutlu embedding'e donustur.
    Encode gait command into low-dimensional embedding.
    """

    def __init__(self, command_dim: int = 6, embed_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(command_dim, 16),
            nn.Tanh(),
            nn.Linear(16, embed_dim),
            nn.Tanh(),
        )

    def forward(self, command: torch.Tensor) -> torch.Tensor:
        return self.net(command)


class HierarchicalLegActor(nn.Module):
    """
    Gait komutuyla kosullanmis bacak kontrolcusu (worker).
    Leg controller conditioned on gait command (worker).

    Girdi: local_obs(18) + gait_embed(8) = 26 -> 3 eklem torku
    """

    def __init__(
        self,
        local_obs_dim: int = 18,
        gait_embed_dim: int = 8,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()

        input_dim = local_obs_dim + gait_embed_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs_with_gait: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log_prob, entropy."""
        hidden = self.net(obs_with_gait)
        mean = self.mean_head(hidden)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# =============================================================================
# Iletisim Aglari / Communication Networks
# =============================================================================

class CommunicationModule(nn.Module):
    """
    Robot-arasi iletisim modulu (CommNet tarzinda).
    Inter-robot communication module (CommNet style).

    Her agent gizli durumundan mesaj uretir,
    diger agentlarin mesajlarini okur ve isler.
    """

    def __init__(self, hidden_dim: int = 64, message_dim: int = 16, num_agents: int = 2):
        super().__init__()
        self.num_agents = num_agents
        self.message_dim = message_dim

        # Mesaj kodlayici / Message encoder: hidden -> message
        self.message_encoder = nn.Sequential(
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh(),
        )

        # Mesaj cozucu / Message decoder: received_messages -> signal
        # Diger agentlarin mesajlarini isler
        self.message_decoder = nn.Sequential(
            nn.Linear(message_dim * (num_agents - 1), hidden_dim),
            nn.Tanh(),
        )

    def encode_message(self, hidden: torch.Tensor) -> torch.Tensor:
        """Gizli durumdan mesaj uret / Produce message from hidden state."""
        return self.message_encoder(hidden)

    def decode_messages(self, received_messages: List[torch.Tensor]) -> torch.Tensor:
        """
        Alinan mesajlari coz / Decode received messages.

        Args:
            received_messages: Diger agentlardan alinan mesajlar

        Returns:
            Communication signal to add to hidden state
        """
        concat = torch.cat(received_messages, dim=-1)
        return self.message_decoder(concat)


class CommActorCritic(nn.Module):
    """
    Iletisim destekli multi-robot agent agi.
    Communication-enabled multi-robot agent network.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 12,
        global_obs_dim: int = 78,
        message_dim: int = 16,
        hidden_dim: int = 256,
        num_agents: int = 2,
    ):
        super().__init__()

        # Gozlem kodlayici / Observation encoder
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Iletisim modulu / Communication module
        self.comm = CommunicationModule(hidden_dim, message_dim, num_agents)

        # Actor (iletisim sonrasi) / Actor (post-communication)
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Centralized critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Gozlemi gizli duruma kodla / Encode observation to hidden state."""
        return self.obs_encoder(obs)

    def produce_message(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mesaj uret / Produce message."""
        return self.comm.encode_message(hidden)

    def act(
        self,
        hidden: torch.Tensor,
        comm_signal: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iletisim sonrasi aksiyon sec / Act after communication."""
        augmented = hidden + comm_signal  # Residual connection
        actor_hidden = self.actor_net(augmented)
        mean = self.actor_mean(actor_hidden)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def get_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Centralized critic degeri / Centralized critic value."""
        return self.critic(global_obs).squeeze(-1)
