import copy
from typing import Any, Dict, List, Optional, Type, Union, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

# from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.policies import ContinuousCritic as NormalContinuousCritic, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    # create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from .extractors import create_mlp, create_cnn
from torch.distributions import Normal
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution as SB_SquashedDiagGaussianDistribution, TanhBijector
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, DiagGaussianDistribution, SelfSquashedDiagGaussianDistribution


from stable_baselines3.sac.policies import Actor as SAC_Actor
from stable_baselines3.sac.policies import SACPolicy
from .extractors import *
from ..type import TensorDict

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -10

class SquashedDiagGaussianDistribution(SB_SquashedDiagGaussianDistribution):
    def proba_distribution(
        self: SelfSquashedDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= (th.log(1 - actions**2 + self.epsilon))
        log_prob = log_prob.sum(dim=-1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        # return None
        return self.distribution.entropy().sum(dim=-1)


def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, (np.ndarray, th.Tensor)):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


class ContinuousCritic(NormalContinuousCritic):
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
            bn: bool = False,
            ln: bool = False,
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
            share_features_extractor
        )
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(n_critics):
            net = create_mlp(input_dim=features_dim + action_dim,
                             output_dim=1,
                             activation_fn=activation_fn,
                             layer=net_arch,
                             bn=bn,
                             ln=ln,
            )
            q_net = nn.Sequential(net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        obs = obs_as_tensor(obs, device=self.device)
        actions = th.as_tensor(actions, device=self.device)
        with th.set_grad_enabled(not self.share_features_extractor):
            if hasattr(self.features_extractor, "recurrent_extractor"):
                features, h = self.extract_features(obs, self.features_extractor)
            else:
                features, h = self.extract_features(obs, self.features_extractor), None
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features, h = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class Actor(SAC_Actor):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = False,
            deterministic: bool = False,
            squash_output: bool = True,
            ln: bool = False,
            bn: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images
        )
        self._squash_output = squash_output
        # Deterministic action
        self.deterministic = deterministic
        self.latent_pi = create_mlp(input_dim=features_dim, layer=net_arch, activation_fn=activation_fn, squash_output=False, bn=bn, ln=ln)
        # self.latent_pi = nn.Flatten()
        self.log_latent_pi = copy.deepcopy(self.latent_pi)
        # self.mu = create_mlp(input_dim=features_dim, layer=net_arch, activation_fn=activation_fn, output_dim=self.action_space.shape[0], squash_output=False)
        # self.log_std = create_mlp(input_dim=features_dim, layer=net_arch, activation_fn=activation_fn, output_dim=self.action_space.shape[0], squash_output=False)
        action_dim = get_action_dim(self.action_space)
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=self.squash_output
            )
        else:
            if self.squash_output:
                self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            else:
                self.action_dist = DiagGaussianDistribution(action_dim)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        deterministic = self.deterministic if deterministic is None else deterministic

        mean_actions, log_std, kwargs, h = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs), h

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor], th.Tensor]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        # obs = obs_as_tensor(obs, device=self.device)
        obs = obs.to(self.device)
        if hasattr(self.features_extractor, "recurrent_extractor"):
            features, h = self.extract_features(obs, self.features_extractor)
        else:
            features, h = self.extract_features(obs, self.features_extractor), None

        latent_pi = self.latent_pi(features)
        log_latent_pi = self.log_latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(log_latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}, h

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # return action and associated log prob
        mean_actions, log_std, kwargs, h = self.get_action_dist_params(obs)
        return *self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs), h

    def get_dist(self, obs):
        mean_actions, log_std, kwargs, h = self.get_action_dist_params(obs)
        return self.action_dist.proba_distribution(mean_actions, log_std, **kwargs), h


class MTDPolicy(SACPolicy):
    features_extractor_alias = {
        "EmptyExtractor": EmptyExtractor,
        "FlattenExtractor": FlattenExtractor,
        "combined": CombinedExtractor,
        "cnn": NatureCNN,
        "StateExtractor": StateExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
        "StateGateExtractor": StateGateExtractor,
    }
    activation_fn_alias = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU
    }

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
            bn: bool = False,
            ln: bool = False,
    ):
        self.ln, self.bn = ln, bn
        self.deterministic = deterministic
        self.__squash_output = squash_output
        if isinstance(features_extractor_class, str):
            features_extractor_class = self.features_extractor_alias[features_extractor_class]

        if isinstance(activation_fn, str):
            activation_fn = self.activation_fn_alias[activation_fn]

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
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["deterministic"] = self.deterministic
        actor_kwargs["squash_output"] = self.__squash_output
        actor_kwargs["ln"] = self.ln
        actor_kwargs["bn"] = self.bn

        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["ln"] = self.ln
        critic_kwargs["bn"] = self.bn
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]], th.Tensor]:
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # obs_tensor = obs_as_tensor(observation, device=self.device)
        # obs_tensor = observation if
        obs_tensor = observation.to(self.device)
        with th.no_grad():
            actions, h = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        return actions, state  # type: ignore[return-value]


MlpPolicy = MTDPolicy


class CnnPolicy(MTDPolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

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
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            deterministic: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            deterministic
        )


class MultiInputPolicy(MTDPolicy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

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
            bn: bool = False,
            ln: bool = False
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            deterministic,
            squash_output,
            bn,
            ln
        )
