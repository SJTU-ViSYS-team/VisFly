from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
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


class Actor(BasePolicy):
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
            normalize_images: bool = True,
            deterministic: bool = True,
            squash_output: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, net_arch, action_dim, activation_fn, squash_output=squash_output)

        # Deterministic action
        self.deterministic = deterministic
        self.mu = nn.Sequential(*actor_net)
        if not deterministic:
            self.sigma = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def get_action(self, obs: th.Tensor, rand_factor=0.) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        mu = self.mu(features)
        if self.deterministic:
            return mu + th.randn_like(mu, device=self.device).clip(min=-1, max=1) * rand_factor
        else:
            return mu + th.randn(mu.shape, device=self.device) * self.sigma(features) * rand_factor

    def forward(self, obs: th.Tensor, rand_factor=1.) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        #   Predictions are always deterministic.
        return self(observation)


class TD3Policy(BasePolicy):
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

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            pi_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            pi_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            vf_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            vf_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            deterministic: bool = True,
            squash_output: bool = False,
    ):
        if features_extractor_class is None:
            assert pi_features_extractor_class is not None and vf_features_extractor_class is not None
            features_extractor_class = pi_features_extractor_class
            features_extractor_kwargs = pi_features_extractor_kwargs
            assert not share_features_extractor, "Shared features extractor is not supported when different extractors are used"
        else:
            assert pi_features_extractor_class is None and vf_features_extractor_class is None
            pi_features_extractor_class = features_extractor_class
            pi_features_extractor_kwargs = features_extractor_kwargs
            vf_features_extractor_class = features_extractor_class
            vf_features_extractor_kwargs = features_extractor_kwargs

        self.pi_features_extractor_class = pi_features_extractor_class
        self.pi_features_extractor_kwargs = pi_features_extractor_kwargs
        self.vf_features_extractor_class = vf_features_extractor_class
        self.vf_features_extractor_kwargs = vf_features_extractor_kwargs

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.deterministic = deterministic
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "deterministic": self.deterministic,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                "squash_output": squash_output,
            }
        )
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

        self._init()

    def _init(self):
        def get_all_linear_submodules(father_module):
            linear_submodules = []
            for submodule in father_module.children():
                if isinstance(submodule, nn.Linear):
                    linear_submodules.append(submodule)
                else:
                    linear_submodules.extend(get_all_linear_submodules(submodule))
            return linear_submodules

        # He Kaiming Initialization
        for net in [self.actor, self.critic, ]:
            # list all the extractors
            for module in get_all_linear_submodules(net):
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=0.0)
                    nn.init.constant_(module.bias, 0.0)

        # copy the weight of the actor to the target actor
        self.actor_target.load_state_dict(self.actor.state_dict())
        # copy the weight of the critic to the target critic
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        if features_extractor is None:
            features_extractor = self.pi_features_extractor_class(self.observation_space, **self.pi_features_extractor_kwargs)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        if features_extractor is None:
            features_extractor = self.vf_features_extractor_class(self.observation_space, **self.vf_features_extractor_kwargs)
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.pop("deterministic")
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic), self.critic(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TD3Policy


class CnnPolicy(TD3Policy):
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
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(TD3Policy):
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
            observation_space: spaces.Dict,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            pi_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            pi_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            vf_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            vf_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            deterministic: bool = True,
            squash_output: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            pi_features_extractor_class,
            pi_features_extractor_kwargs,
            vf_features_extractor_class,
            vf_features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            deterministic,
            squash_output
        )
