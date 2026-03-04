"""
Egitim Callback'leri / Training Callbacks

Egitim sirasinda reward takibi, checkpoint kaydetme ve
degerlendirme islemleri icin callback siniflari.

Callback classes for reward logging, checkpoint saving,
and evaluation during training.
"""

import numpy as np
import torch
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Egitim gecmisini depola / Store training history."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    timestep_rewards: List[float] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_timesteps: List[int] = field(default_factory=list)
    extra_metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_episode(self, reward: float, length: int, timestep: int):
        """Episode sonucunu ekle / Add episode result."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.timestep_rewards.append(reward)
        self.timesteps.append(timestep)

    def add_eval(self, reward: float, timestep: int):
        """Degerlendirme sonucunu ekle / Add eval result."""
        self.eval_rewards.append(reward)
        self.eval_timesteps.append(timestep)

    def add_metric(self, name: str, value: float):
        """Ek metrik ekle / Add extra metric."""
        if name not in self.extra_metrics:
            self.extra_metrics[name] = []
        self.extra_metrics[name].append(value)

    @property
    def mean_reward_last_100(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards[-100:]))


class RewardLogger:
    """
    Episode reward ve uzunluk takip eden logger.
    Logger that tracks episode rewards and lengths.
    """

    def __init__(self, print_interval: int = 10, agent_name: str = ""):
        self.print_interval = print_interval
        self.agent_name = agent_name
        self.history = TrainingHistory()
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._total_timesteps = 0

    def on_step(self, reward: float, done: bool):
        """Her adimda cagir / Call at each step."""
        self._current_episode_reward += reward
        self._current_episode_length += 1
        self._total_timesteps += 1

        if done:
            self.history.add_episode(
                self._current_episode_reward,
                self._current_episode_length,
                self._total_timesteps,
            )

            n_episodes = len(self.history.episode_rewards)
            if n_episodes % self.print_interval == 0:
                avg = self.history.mean_reward_last_100
                prefix = f"[{self.agent_name}] " if self.agent_name else ""
                print(
                    f"  {prefix}Episode {n_episodes:>5d} | "
                    f"Timestep {self._total_timesteps:>8d} | "
                    f"Avg Reward (100): {avg:>8.2f}"
                )

            self._current_episode_reward = 0.0
            self._current_episode_length = 0

    @property
    def total_timesteps(self) -> int:
        return self._total_timesteps


class BestModelCallback:
    """
    En iyi modeli takip edip kaydeden callback.
    Tracks and saves the best model, deletes old checkpoints.

    - Periyodik olarak mevcut performansi kontrol eder (son N episode ortalamasi)
    - Daha iyi bir model bulundugunda best_model.pt olarak kaydeder
    - Eski checkpoint dosyalarini siler
    - Egitim sonunda sadece best_model.pt kalir
    """

    def __init__(
        self,
        save_dir: str,
        agent_name: str = "",
        eval_window: int = 100,
        check_interval: int = 10_000,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name
        self.eval_window = eval_window
        self.check_interval = check_interval

        self.best_reward = -float("inf")
        self.best_timestep = 0
        self._last_check_timestep = 0
        self._prev_checkpoint_path: Optional[Path] = None

    def on_step(
        self,
        timestep: int,
        model_dict: dict,
        logger: RewardLogger,
    ) -> bool:
        """
        Performansi kontrol et, gerekirse kaydet.
        Check performance, save if improved.

        Returns:
            True eger yeni best model kaydedildiyse
        """
        if timestep - self._last_check_timestep < self.check_interval:
            return False

        self._last_check_timestep = timestep

        # Yeterli episode yoksa atla
        if len(logger.history.episode_rewards) < 10:
            return False

        current_avg = logger.history.mean_reward_last_100

        if current_avg > self.best_reward:
            old_best = self.best_reward
            self.best_reward = current_avg
            self.best_timestep = timestep

            # Eski checkpoint'i sil / Delete old checkpoint
            if self._prev_checkpoint_path and self._prev_checkpoint_path.exists():
                self._prev_checkpoint_path.unlink()

            # Yeni best model kaydet / Save new best model
            best_path = self.save_dir / "best_model.pt"
            torch.save({
                "model": model_dict,
                "reward": current_avg,
                "timestep": timestep,
            }, str(best_path))
            self._prev_checkpoint_path = best_path

            prefix = f"[{self.agent_name}] " if self.agent_name else ""
            print(
                f"  {prefix}** Yeni best model! "
                f"Reward: {old_best:.2f} -> {current_avg:.2f} "
                f"(timestep {timestep:,}) **"
            )
            return True

        return False

    def save_final(self, model_dict: dict, logger: RewardLogger):
        """
        Egitim sonu: final durumu kaydet.
        End of training: save final state.

        Eger final model best model'den daha iyiyse guncelle,
        degilse best_model.pt'yi koru.
        """
        current_avg = logger.history.mean_reward_last_100

        # Eski checkpoint dosyalarini temizle (best_model.pt haric)
        self._cleanup_old_checkpoints()

        if current_avg >= self.best_reward:
            # Final ayni zamanda best -> best_model.pt guncelle
            self.best_reward = current_avg
            best_path = self.save_dir / "best_model.pt"
            torch.save({
                "model": model_dict,
                "reward": current_avg,
                "timestep": logger.total_timesteps,
            }, str(best_path))
            print(
                f"  Model kaydedildi: {best_path} "
                f"(reward: {current_avg:.2f})"
            )
        else:
            # Best model zaten daha iyi, dokunma
            best_path = self.save_dir / "best_model.pt"
            print(
                f"  Best model korundu: {best_path} "
                f"(best: {self.best_reward:.2f}, final: {current_avg:.2f})"
            )

    def _cleanup_old_checkpoints(self):
        """best_model.pt disindaki tum checkpoint'leri sil."""
        for f in self.save_dir.glob("checkpoint_*.pt"):
            f.unlink()
            print(f"  Eski checkpoint silindi: {f.name}")

    def get_best_reward(self) -> float:
        return self.best_reward

    @staticmethod
    def load_best(save_dir: str) -> Optional[dict]:
        """
        Kaydedilmis best model'i yukle / Load saved best model.

        Returns:
            model_dict veya None
        """
        best_path = Path(save_dir) / "best_model.pt"
        if best_path.exists():
            data = torch.load(str(best_path), weights_only=False)
            print(
                f"  Best model yuklendi: {best_path} "
                f"(reward: {data['reward']:.2f}, timestep: {data['timestep']:,})"
            )
            return data["model"]
        return None
