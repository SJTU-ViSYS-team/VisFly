from gymnasium.vector.utils import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

from .extractors import create_mlp
from .td_policies import MTDPolicy, MultiInputPolicy, obs_as_tensor
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.policies import ContinuousCritic as NormalContinuousCritic
from copy import deepcopy

class NoActorContinuousCritic(NormalContinuousCritic):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            bn=False,
            ln=False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,

        )
        self.q_networks = []
        for idx in range(n_critics):
            net = create_mlp(input_dim=features_dim,
                             output_dim=1,
                             activation_fn=activation_fn,
                             layer=net_arch,
                             bn=bn,
                             ln=ln,
            )
            q_net = nn.Sequential(net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        obs = obs_as_tensor(obs, device=self.device)
        with th.set_grad_enabled(not self.share_features_extractor):
            if hasattr(self.features_extractor, "recurrent_extractor"):
                features, h = self.extract_features(obs, self.features_extractor)
            else:
                features, h = self.extract_features(obs, self.features_extractor), None
        qvalue_input = features
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features, h = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features)


class DMPolicy(MTDPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            deterministic=False,
            squash_output=True,
            bn=False,
            ln=False
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            deterministic=deterministic,
            squash_output=squash_output,
            bn=bn,
            ln=ln
        )
        self.actor_target = deepcopy(self.actor)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> NoActorContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["ln"] = self.ln
        critic_kwargs["bn"] = self.bn
        return NoActorContinuousCritic(**critic_kwargs).to(self.device)


class MultiInputDMPolicy(DMPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            deterministic=False,
            squash_output=True,
            bn=False,
            ln=False
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            deterministic=deterministic,
            squash_output=squash_output,
            bn=bn,
            ln=ln
        )

