"""
Egitim Dongusu / Training Loop

Farkli agent turleri icin egitim yoneticisi siniflari.
Trainer classes for different agent types.

Her trainer ortak arayuz kullanir:
    trainer = XxxTrainer(env_config, agent_config)
    history = trainer.train(total_timesteps)
    trainer.evaluate(num_episodes, render)
"""

import numpy as np
import torch
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EnvConfig, PPOConfig, MAPPOConfig, HierarchicalConfig, MultiRobotConfig, SAVE_DIR
from envs.quadruped_env import UnitreeQuadrupedEnv
from envs.multi_robot_env import MultiRobotQuadrupedEnv
from training.buffer import RolloutBuffer, MultiAgentRolloutBuffer, compute_gae
from training.callbacks import RewardLogger, BestModelCallback, TrainingHistory


# =============================================================================
# Single-Agent PPO Trainer
# =============================================================================

class SingleAgentTrainer:
    """Tek-agent PPO ile quadruped egitimi / Single-agent PPO quadruped training."""

    def __init__(self, env_config: EnvConfig, ppo_config: PPOConfig):
        self.env_config = env_config
        self.config = ppo_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy import to avoid circular dependency
        from agents.ppo_agent import PPOAgent

        self.env = UnitreeQuadrupedEnv(
            xml_file=env_config.xml_path,
            frame_skip=env_config.frame_skip,
            forward_reward_weight=env_config.forward_reward_weight,
            ctrl_cost_weight=env_config.ctrl_cost_weight,
            healthy_reward=env_config.healthy_reward,
            energy_cost_weight=env_config.energy_cost_weight,
            terminate_when_unhealthy=env_config.terminate_when_unhealthy,
            healthy_z_range=env_config.healthy_z_range,
            reset_noise_scale=env_config.reset_noise_scale,
        )

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=ppo_config.hidden_dim,
            lr=ppo_config.lr,
            clip_eps=ppo_config.clip_eps,
            entropy_coeff=ppo_config.entropy_coeff,
            value_loss_coeff=ppo_config.value_loss_coeff,
            max_grad_norm=ppo_config.max_grad_norm,
            device=self.device,
        )

        self.buffer = RolloutBuffer(
            buffer_size=ppo_config.rollout_length,
            obs_dim=obs_dim,
            action_dim=act_dim,
        )

    def train(self, total_timesteps: int = None) -> TrainingHistory:
        """PPO egitimi calistir / Run PPO training."""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        print("\n" + "=" * 60)
        print("  SINGLE-AGENT PPO EGITIMI / TRAINING")
        print(f"  Timesteps: {total_timesteps:,}")
        print("=" * 60)

        logger = RewardLogger(print_interval=10, agent_name="PPO")
        best_cb = BestModelCallback(
            save_dir=str(SAVE_DIR / "single_ppo"),
            agent_name="PPO",
            check_interval=self.config.eval_interval,
        )

        obs, _ = self.env.reset()
        timestep = 0

        while timestep < total_timesteps:
            # Rollout topla / Collect rollout
            self.buffer.reset()
            for _ in range(self.config.rollout_length):
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, value, log_prob, done)
                logger.on_step(reward, done)
                timestep += 1

                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

                if self.buffer.is_full:
                    break

            # GAE hesapla / Compute GAE
            with torch.no_grad():
                _, _, next_value = self.agent.select_action(obs)
            self.buffer.compute_returns_and_advantages(
                next_value, self.config.gamma, self.config.gae_lambda
            )

            # Best model kontrolu: update ONCESI agirliklar kaydedilmeli
            pre_update_state = self.agent.get_state_dict()
            best_cb.on_step(timestep, pre_update_state, logger)

            # PPO guncelle / PPO update
            for _ in range(self.config.ppo_epochs):
                for batch in self.buffer.get_minibatches(
                    self.config.minibatch_size, self.device
                ):
                    self.agent.update(*batch)

        best_cb.save_final(self.agent.get_state_dict(), logger)
        self.env.close()

        print(f"\n  Egitim tamamlandi! Best reward: {best_cb.get_best_reward():.2f}")
        return logger.history

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """Egitilmis agent'i degerlendir / Evaluate trained agent."""
        render_mode = "human" if render else None
        env = UnitreeQuadrupedEnv(
            xml_file=self.env_config.xml_path,
            frame_skip=self.env_config.frame_skip,
            render_mode=render_mode,
        )

        rewards = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        env.close()
        mean_reward = float(np.mean(rewards))
        print(f"  Eval ({num_episodes} ep): {mean_reward:.2f} +/- {np.std(rewards):.2f}")
        return mean_reward


# =============================================================================
# MAPPO Trainer
# =============================================================================

class MAPPOTrainer:
    """Multi-Agent PPO (per-leg CTDE) egitimi / MAPPO training."""

    def __init__(self, env_config: EnvConfig, mappo_config: MAPPOConfig):
        self.env_config = env_config
        self.config = mappo_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from agents.mappo_agent import MAPPOController

        self.env = UnitreeQuadrupedEnv(
            xml_file=env_config.xml_path,
            frame_skip=env_config.frame_skip,
            forward_reward_weight=env_config.forward_reward_weight,
            ctrl_cost_weight=env_config.ctrl_cost_weight,
            healthy_reward=env_config.healthy_reward,
            energy_cost_weight=env_config.energy_cost_weight,
            terminate_when_unhealthy=env_config.terminate_when_unhealthy,
            healthy_z_range=env_config.healthy_z_range,
            reset_noise_scale=env_config.reset_noise_scale,
        )

        self.controller = MAPPOController(
            num_agents=mappo_config.num_agents,
            local_obs_dim=mappo_config.local_obs_dim,
            global_obs_dim=mappo_config.global_obs_dim,
            action_dim=mappo_config.action_dim_per_agent,
            actor_hidden=mappo_config.actor_hidden_dim,
            critic_hidden=mappo_config.critic_hidden_dim,
            lr_actor=mappo_config.lr_actor,
            lr_critic=mappo_config.lr_critic,
            clip_eps=mappo_config.clip_eps,
            entropy_coeff=mappo_config.entropy_coeff,
            max_grad_norm=mappo_config.max_grad_norm,
            device=self.device,
        )

        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=mappo_config.rollout_length,
            num_agents=mappo_config.num_agents,
            local_obs_dim=mappo_config.local_obs_dim,
            global_obs_dim=mappo_config.global_obs_dim,
            action_dim_per_agent=mappo_config.action_dim_per_agent,
        )

    def train(self, total_timesteps: int = None) -> TrainingHistory:
        """MAPPO egitimi / MAPPO training."""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        print("\n" + "=" * 60)
        print("  MAPPO (PER-LEG CTDE) EGITIMI / TRAINING")
        print(f"  Agents: {self.config.num_agents} | Timesteps: {total_timesteps:,}")
        print("=" * 60)

        logger = RewardLogger(print_interval=10, agent_name="MAPPO")
        best_cb = BestModelCallback(
            save_dir=str(SAVE_DIR / "mappo"),
            agent_name="MAPPO",
            check_interval=self.config.eval_interval,
        )

        # Linear LR decay icin baslangic degerleri
        initial_lr_actor = self.config.lr_actor
        initial_lr_critic = self.config.lr_critic

        obs, _ = self.env.reset()
        timestep = 0

        while timestep < total_timesteps:
            # Linear LR decay: egitim ilerledikce lr kucult
            progress = timestep / total_timesteps
            current_lr_actor = max(initial_lr_actor * (1.0 - progress), 1e-5)
            current_lr_critic = max(initial_lr_critic * (1.0 - progress), 1e-5)
            for agent_idx in range(self.config.num_agents):
                for param_group in self.controller.agents[agent_idx].optimizer.param_groups:
                    param_group["lr"] = current_lr_actor
            for param_group in self.controller.critic_optimizer.param_groups:
                param_group["lr"] = current_lr_critic

            self.buffer.reset()
            for _ in range(self.config.rollout_length):
                global_obs = self.env.get_global_state()
                local_obs_list = [self.env.get_leg_obs(i) for i in range(self.config.num_agents)]

                # Her agent aksiyon sec / Each agent selects action
                actions_list, log_probs_list = self.controller.select_actions(local_obs_list)

                # Centralized critic degeri / Centralized critic value
                value = self.controller.get_value(global_obs)

                # Aksiyonlari birlestir / Concatenate actions
                full_action = np.concatenate(actions_list)
                next_obs, reward, terminated, truncated, info = self.env.step(full_action)
                done = terminated or truncated

                self.buffer.add(
                    local_obs_list, global_obs, actions_list,
                    reward, value, log_probs_list, done,
                )

                logger.on_step(reward, done)
                timestep += 1

                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

                if self.buffer.is_full:
                    break

            # GAE hesapla / Compute GAE
            next_global_obs = self.env.get_global_state()
            next_value = self.controller.get_value(next_global_obs)
            self.buffer.compute_returns_and_advantages(
                next_value, self.config.gamma, self.config.gae_lambda
            )

            # Best model kontrolu: update ONCESI agirliklar kaydedilmeli
            # cunku reward bu agirliklarla elde edildi
            pre_update_state = self.controller.get_state_dict()
            best_cb.on_step(timestep, pre_update_state, logger)

            # MAPPO guncelle / MAPPO update
            for _ in range(self.config.ppo_epochs):
                # Critic guncelle (global obs ile)
                for batch in self.buffer.get_minibatches(
                    self.config.minibatch_size, device=self.device
                ):
                    self.controller.update_critic(batch[1], batch[5])  # global_obs, returns

                # Her actor guncelle (local obs ile)
                for agent_idx in range(self.config.num_agents):
                    for batch in self.buffer.get_minibatches(
                        self.config.minibatch_size, agent_idx=agent_idx, device=self.device
                    ):
                        self.controller.update_actor(
                            agent_idx, batch[0], batch[2], batch[3], batch[4]
                        )  # local_obs, actions, old_log_probs, advantages

        best_cb.save_final(self.controller.get_state_dict(), logger)
        self.env.close()

        print(f"\n  Egitim tamamlandi! Best reward: {best_cb.get_best_reward():.2f}")
        return logger.history

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """MAPPO degerlendir / Evaluate MAPPO."""
        render_mode = "human" if render else None
        env = UnitreeQuadrupedEnv(
            xml_file=self.env_config.xml_path,
            frame_skip=self.env_config.frame_skip,
            render_mode=render_mode,
        )

        rewards = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done:
                local_obs_list = [env.get_leg_obs(i) for i in range(self.config.num_agents)]
                actions_list, _ = self.controller.select_actions(local_obs_list, deterministic=True)
                full_action = np.concatenate(actions_list)
                obs, reward, terminated, truncated, _ = env.step(full_action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            rewards.append(total_reward)
            print(f"    Episode {ep+1}/{num_episodes}: reward={total_reward:.2f}, steps={steps}")

        env.close()
        mean_reward = float(np.mean(rewards))
        print(f"  Eval ({num_episodes} ep): {mean_reward:.2f} +/- {np.std(rewards):.2f}")
        return mean_reward


# =============================================================================
# Hierarchical Trainer
# =============================================================================

class HierarchicalTrainer:
    """Hiyerarsik manager-worker egitimi / Hierarchical training."""

    def __init__(self, env_config: EnvConfig, hier_config: HierarchicalConfig):
        self.env_config = env_config
        self.config = hier_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from agents.hierarchical_agent import HierarchicalController

        self.env = UnitreeQuadrupedEnv(
            xml_file=env_config.xml_path,
            frame_skip=env_config.frame_skip,
            forward_reward_weight=env_config.forward_reward_weight,
            ctrl_cost_weight=env_config.ctrl_cost_weight,
            healthy_reward=env_config.healthy_reward,
            energy_cost_weight=env_config.energy_cost_weight,
            terminate_when_unhealthy=env_config.terminate_when_unhealthy,
            healthy_z_range=env_config.healthy_z_range,
            reset_noise_scale=env_config.reset_noise_scale,
        )

        self.controller = HierarchicalController(
            manager_obs_dim=hier_config.manager_obs_dim,
            manager_action_dim=hier_config.manager_action_dim,
            manager_hidden=hier_config.manager_hidden_dim,
            worker_obs_dim=hier_config.worker_obs_dim,
            worker_action_dim=hier_config.worker_action_dim,
            worker_hidden=hier_config.worker_hidden_dim,
            gait_embed_dim=hier_config.gait_embed_dim,
            lr_manager=hier_config.lr_manager,
            lr_worker=hier_config.lr_worker,
            clip_eps=hier_config.clip_eps,
            entropy_coeff=hier_config.entropy_coeff,
            max_grad_norm=hier_config.max_grad_norm,
            decision_period=hier_config.manager_decision_period,
            device=self.device,
        )

    def train(self, total_timesteps: int = None) -> TrainingHistory:
        """Hiyerarsik egitim / Hierarchical training."""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        print("\n" + "=" * 60)
        print("  HIERARCHICAL (MANAGER-WORKER) EGITIMI / TRAINING")
        print(f"  Manager Period: {self.config.manager_decision_period} | Timesteps: {total_timesteps:,}")
        print("=" * 60)

        logger = RewardLogger(print_interval=10, agent_name="Hierarchical")
        best_cb = BestModelCallback(
            save_dir=str(SAVE_DIR / "hierarchical"),
            agent_name="Hierarchical",
            check_interval=self.config.eval_interval,
        )

        K = self.config.manager_decision_period
        obs, _ = self.env.reset()
        timestep = 0

        # Manager ve worker icin ayri buffer'lar
        manager_obs_list = []
        manager_actions_list = []
        manager_log_probs_list = []
        manager_rewards_list = []
        manager_values_list = []
        manager_dones_list = []

        worker_buffer = MultiAgentRolloutBuffer(
            buffer_size=self.config.rollout_length,
            num_agents=4,
            local_obs_dim=self.config.worker_obs_dim,
            global_obs_dim=39,  # full robot obs
            action_dim_per_agent=self.config.worker_action_dim,
        )

        while timestep < total_timesteps:
            worker_buffer.reset()
            manager_obs_list.clear()
            manager_actions_list.clear()
            manager_log_probs_list.clear()
            manager_rewards_list.clear()
            manager_values_list.clear()
            manager_dones_list.clear()

            rollout_steps = 0
            while rollout_steps < self.config.rollout_length:
                # Manager karar noktasi / Manager decision
                torso_state = self.env.get_torso_state()
                x_vel = self.env.data.qvel[0]
                y_vel = self.env.data.qvel[1]
                phase = np.array([np.sin(timestep * 0.01), np.cos(timestep * 0.01)])
                manager_obs = np.concatenate([torso_state, [x_vel, y_vel], phase])

                gait_cmd, m_log_prob, m_value = self.controller.manager_action(manager_obs)
                manager_obs_list.append(manager_obs)
                manager_actions_list.append(gait_cmd)
                manager_log_probs_list.append(m_log_prob)
                manager_values_list.append(m_value)

                # Worker'lar K adim boyunca calisir
                manager_reward_accum = 0.0
                manager_done = False
                gait_embed = self.controller.encode_gait_command(gait_cmd)

                for k_step in range(K):
                    global_obs = self.env.get_global_state()
                    local_obs_list = []
                    for i in range(4):
                        leg_obs = self.env.get_leg_obs(i)
                        worker_obs = np.concatenate([leg_obs, gait_embed])
                        local_obs_list.append(worker_obs)

                    actions_list, w_log_probs = self.controller.worker_actions(local_obs_list)
                    w_value = self.controller.get_worker_value(global_obs)

                    full_action = np.concatenate(actions_list)
                    next_obs, reward, terminated, truncated, info = self.env.step(full_action)
                    done = terminated or truncated

                    worker_buffer.add(
                        local_obs_list, global_obs, actions_list,
                        reward, w_value, w_log_probs, done,
                    )

                    manager_reward_accum += reward
                    logger.on_step(reward, done)
                    timestep += 1
                    rollout_steps += 1

                    if done:
                        manager_done = True
                        obs, _ = self.env.reset()
                        break
                    else:
                        obs = next_obs

                manager_rewards_list.append(manager_reward_accum)
                manager_dones_list.append(manager_done)

                if rollout_steps >= self.config.rollout_length:
                    break

            # Best model kontrolu: update ONCESI agirliklar kaydedilmeli
            pre_update_state = self.controller.get_state_dict()
            best_cb.on_step(timestep, pre_update_state, logger)

            # Worker GAE & update
            if worker_buffer.ptr > 0:
                next_global = self.env.get_global_state()
                next_w_value = self.controller.get_worker_value(next_global)
                worker_buffer.compute_returns_and_advantages(
                    next_w_value, self.config.gamma, self.config.gae_lambda
                )

                for _ in range(self.config.ppo_epochs):
                    for batch in worker_buffer.get_minibatches(
                        self.config.minibatch_size, device=self.device
                    ):
                        self.controller.update_workers(batch)

            # Manager GAE & update
            if len(manager_rewards_list) > 1:
                m_rewards = np.array(manager_rewards_list, dtype=np.float32)
                m_values = np.array(manager_values_list, dtype=np.float32)
                m_dones = np.array(manager_dones_list, dtype=np.float32)

                # Basit manager GAE
                m_adv, m_ret = compute_gae(
                    m_rewards, m_values, m_values[-1], m_dones,
                    self.config.gamma ** K, self.config.gae_lambda,
                )

                self.controller.update_manager(
                    np.array(manager_obs_list),
                    np.array(manager_actions_list),
                    np.array(manager_log_probs_list),
                    m_adv, m_ret,
                    self.device,
                )

        best_cb.save_final(self.controller.get_state_dict(), logger)
        self.env.close()

        print(f"\n  Egitim tamamlandi! Best reward: {best_cb.get_best_reward():.2f}")
        return logger.history

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """Hiyerarsik controller degerlendir / Evaluate hierarchical controller."""
        render_mode = "human" if render else None
        env = UnitreeQuadrupedEnv(
            xml_file=self.env_config.xml_path,
            frame_skip=self.env_config.frame_skip,
            render_mode=render_mode,
        )
        K = self.config.manager_decision_period

        rewards = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            while not done:
                torso_state = env.get_torso_state()
                phase = np.array([np.sin(step_count * 0.01), np.cos(step_count * 0.01)])
                manager_obs = np.concatenate([torso_state, [env.data.qvel[0], env.data.qvel[1]], phase])
                gait_cmd, _, _ = self.controller.manager_action(manager_obs, deterministic=True)
                gait_embed = self.controller.encode_gait_command(gait_cmd)

                for _ in range(K):
                    local_obs_list = []
                    for i in range(4):
                        leg_obs = env.get_leg_obs(i)
                        local_obs_list.append(np.concatenate([leg_obs, gait_embed]))
                    actions_list, _ = self.controller.worker_actions(local_obs_list, deterministic=True)
                    full_action = np.concatenate(actions_list)
                    obs, reward, terminated, truncated, _ = env.step(full_action)
                    total_reward += reward
                    step_count += 1
                    done = terminated or truncated
                    if done:
                        break
            rewards.append(total_reward)

        env.close()
        mean_reward = float(np.mean(rewards))
        print(f"  Eval ({num_episodes} ep): {mean_reward:.2f} +/- {np.std(rewards):.2f}")
        return mean_reward


# =============================================================================
# Multi-Robot Trainer
# =============================================================================

class MultiRobotTrainer:
    """Coklu robot formation egitimi / Multi-robot formation training."""

    def __init__(self, env_config: EnvConfig, mr_config: MultiRobotConfig):
        self.env_config = env_config
        self.config = mr_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from agents.communication import CommMAPPOAgent

        self.env = MultiRobotQuadrupedEnv(
            num_robots=mr_config.num_robots,
            formation_type=mr_config.formation_type,
            formation_spacing=mr_config.formation_spacing,
            frame_skip=env_config.frame_skip,
            forward_reward_weight=mr_config.forward_weight,
            formation_reward_weight=mr_config.formation_weight,
            ctrl_cost_weight=env_config.ctrl_cost_weight,
            healthy_reward=env_config.healthy_reward,
            healthy_z_range=env_config.healthy_z_range,
        )

        obs_dim = self.env.obs_dim_per_robot
        global_dim = mr_config.num_robots * 39

        self.agents = []
        for i in range(mr_config.num_robots):
            agent = CommMAPPOAgent(
                obs_dim=obs_dim,
                action_dim=12,
                global_obs_dim=global_dim,
                message_dim=mr_config.message_dim,
                hidden_dim=mr_config.hidden_dim,
                num_agents=mr_config.num_robots,
                lr=mr_config.lr,
                clip_eps=mr_config.clip_eps,
                entropy_coeff=mr_config.entropy_coeff,
                max_grad_norm=mr_config.max_grad_norm,
                device=self.device,
            )
            self.agents.append(agent)

    def train(self, total_timesteps: int = None) -> TrainingHistory:
        """Multi-robot egitimi / Multi-robot training."""
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        print("\n" + "=" * 60)
        print("  MULTI-ROBOT FORMATION EGITIMI / TRAINING")
        print(f"  Robots: {self.config.num_robots} | Timesteps: {total_timesteps:,}")
        print("=" * 60)

        logger = RewardLogger(print_interval=10, agent_name="MultiRobot")
        best_cb = BestModelCallback(
            save_dir=str(SAVE_DIR / "multi_robot"),
            agent_name="MultiRobot",
            check_interval=self.config.eval_interval,
        )

        # Basit rollout buffer (per robot)
        obs_list = self.env.reset(seed=self.config.seed)
        timestep = 0

        # Trajectory storage
        batch_obs = {i: [] for i in range(self.config.num_robots)}
        batch_actions = {i: [] for i in range(self.config.num_robots)}
        batch_log_probs = {i: [] for i in range(self.config.num_robots)}
        batch_rewards = []
        batch_values = []
        batch_dones = []
        batch_global_obs = []

        while timestep < total_timesteps:
            # Temizle / Clear
            for i in range(self.config.num_robots):
                batch_obs[i].clear()
                batch_actions[i].clear()
                batch_log_probs[i].clear()
            batch_rewards.clear()
            batch_values.clear()
            batch_dones.clear()
            batch_global_obs.clear()

            for _ in range(self.config.rollout_length):
                global_obs = self.env.get_global_state()

                # Phase 1: Encode + message exchange
                hidden_states = []
                for i in range(self.config.num_robots):
                    h = self.agents[i].encode(obs_list[i])
                    hidden_states.append(h)

                messages = []
                for i in range(self.config.num_robots):
                    msg = self.agents[i].produce_message(hidden_states[i])
                    messages.append(msg)

                # Phase 2: Act with communication
                actions = []
                log_probs = []
                for i in range(self.config.num_robots):
                    other_msgs = [messages[j] for j in range(self.config.num_robots) if j != i]
                    action, lp = self.agents[i].act_with_comm(hidden_states[i], other_msgs)
                    actions.append(action)
                    log_probs.append(lp)

                # Centralized value (ilk agent'in critic'i)
                value = self.agents[0].get_value(global_obs)

                next_obs_list, rewards, done, info = self.env.step(actions)
                team_reward = float(np.mean(rewards))

                for i in range(self.config.num_robots):
                    batch_obs[i].append(obs_list[i].copy())
                    batch_actions[i].append(actions[i].copy())
                    batch_log_probs[i].append(log_probs[i])
                batch_rewards.append(team_reward)
                batch_values.append(value)
                batch_dones.append(float(done))
                batch_global_obs.append(global_obs.copy())

                logger.on_step(team_reward, done)
                timestep += 1

                if done:
                    obs_list = self.env.reset()
                else:
                    obs_list = next_obs_list

            # GAE
            next_global = self.env.get_global_state()
            next_value = self.agents[0].get_value(next_global)
            rewards_arr = np.array(batch_rewards, dtype=np.float32)
            values_arr = np.array(batch_values, dtype=np.float32)
            dones_arr = np.array(batch_dones, dtype=np.float32)
            advantages, returns = compute_gae(
                rewards_arr, values_arr, next_value, dones_arr,
                self.config.gamma, self.config.gae_lambda,
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Best model kontrolu: update ONCESI agirliklar kaydedilmeli
            pre_update_dicts = {f"agent_{i}": self.agents[i].get_state_dict() for i in range(self.config.num_robots)}
            best_cb.on_step(timestep, pre_update_dicts, logger)

            # Update each agent
            for i in range(self.config.num_robots):
                self.agents[i].ppo_update(
                    np.array(batch_obs[i]),
                    np.array(batch_actions[i]),
                    np.array(batch_log_probs[i]),
                    advantages,
                    returns,
                    np.array(batch_global_obs),
                    self.config.ppo_epochs,
                    self.config.minibatch_size,
                )

        final_dicts = {f"agent_{i}": self.agents[i].get_state_dict() for i in range(self.config.num_robots)}
        best_cb.save_final(final_dicts, logger)
        self.env.close()

        print(f"\n  Egitim tamamlandi! Best reward: {best_cb.get_best_reward():.2f}")
        return logger.history

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """Multi-robot degerlendir / Evaluate multi-robot."""
        env = MultiRobotQuadrupedEnv(
            num_robots=self.config.num_robots,
            formation_type=self.config.formation_type,
            formation_spacing=self.config.formation_spacing,
        )

        rewards = []
        for ep in range(num_episodes):
            obs_list = env.reset()
            total_reward = 0
            done = False
            while not done:
                hidden_states = [self.agents[i].encode(obs_list[i]) for i in range(self.config.num_robots)]
                messages = [self.agents[i].produce_message(hidden_states[i]) for i in range(self.config.num_robots)]
                actions = []
                for i in range(self.config.num_robots):
                    other_msgs = [messages[j] for j in range(self.config.num_robots) if j != i]
                    action, _ = self.agents[i].act_with_comm(hidden_states[i], other_msgs, deterministic=True)
                    actions.append(action)
                obs_list, r, done, info = env.step(actions)
                total_reward += np.mean(r)
            rewards.append(total_reward)

        env.close()
        mean_reward = float(np.mean(rewards))
        print(f"  Eval ({num_episodes} ep): {mean_reward:.2f} +/- {np.std(rewards):.2f}")
        return mean_reward
