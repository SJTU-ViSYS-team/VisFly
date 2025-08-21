import io
import os
import pathlib
import sys
import time
from typing import Union, Tuple, Optional, Type, Dict, Any, Iterable

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import Schedule, GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.sac import SAC as sb_SAC
from stable_baselines3.sac.policies import SACPolicy

import torch as th
from stable_baselines3.sac.sac import SelfSAC
from ..policies.sac_polices import SACPolicy, MultiInputPolicy


class SAC(sb_SAC):
    policy_aliases ={
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(self, comment="", save_path=None, *args, **kwargs):
        self.comment = comment
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.policy_save_path = f"{self.save_path}/PPO_{self.comment}" if comment is not None else f"{self.save_path}/ppo"
        kwargs["tensorboard_log"] = self.save_path
        super().__init__(*args, **kwargs)

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name+"_"+self.comment if self.comment is not None else tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

            if len(self.ep_info_buffer[0]["extra"]) >= 0:
                for key in self.ep_info_buffer[0]["extra"].keys():
                    self.logger.record(
                        f"rollout/ep_{key}_mean",
                        safe_mean(
                            [ep_info["extra"][key] for ep_info in self.ep_info_buffer]
                        ),
                    )
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/ep_success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase]=None,
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:

        if path is None:
            path = self.policy_save_path
        super().save(
            path=path,
            exclude=exclude,
            include=include,
        )