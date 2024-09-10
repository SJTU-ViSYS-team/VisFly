import warnings
from functools import partial
import numpy as np

import torch.nn as nn
import torch as th
from typing import List, Optional, Type, Union, ClassVar, Dict, Any, Tuple

from stable_baselines3.common.distributions import make_proba_distribution, DiagGaussianDistribution, StateDependentNoiseDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, Distribution
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs

from utils.policies.extractors import create_mlp, StateExtractor, StateTargetImageExtractor, StateTargetExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, MlpExtractor
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy, BaseModel


class ActorPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        self.log_std_init = log_std_init
        dist_kwargs = None
        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        # del self.mlp_extractor.value_net

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dim = self.action_space.shape[0]
        # self.action_net = nn.Linear(latent_dim_pi, self.action_dim)

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def get_action(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the deterministic action from an observation

        :param obs: Observation
        :return: Deterministic action
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        return self.action_net(latent_pi)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs: th.Tensor, deterministic=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape))

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_action(observation)
    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
    #     """
    #     Retrieve action distribution given the latent codes.
    #
    #     :param latent_pi: Latent code for the actor
    #     :return: Action distribution
    #     """
    #     mean_actions = self.action_net(latent_pi)
    #
    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
    #     else:
    #         raise ValueError("Invalid action distribution")



    # def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
    #     """
    #     Evaluate actions according to the current policy,
    #     given the observations.
    #
    #     :param obs: Observation
    #     :param actions: Actions
    #     :return: estimated value, log likelihood of taking those actions
    #         and entropy of the action distribution.
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     log_prob = distribution.log_prob(actions)
    #     values = self.value_net(latent_vf)
    #     entropy = distribution.entropy()
    #     return values, log_prob, entropy

    # def get_distribution(self, obs: PyTorchObs) -> Distribution:
    #     """
    #     Get the current policy distribution given the observations.
    #
    #     :param obs:
    #     :return: the action distribution.
    #     """
    #     features = super().extract_features(obs, self.pi_features_extractor)
    #     latent_pi = self.mlp_extractor.forward_actor(features)
    #     return self._get_action_dist_from_latent(latent_pi)

    # def predict_values(self, obs: PyTorchObs) -> th.Tensor:
    #     """
    #     Get the estimated values according to the current policy given the observations.
    #
    #     :param obs: Observation
    #     :return: the estimated values.
    #     """
    #     features = super().extract_features(obs, self.vf_features_extractor)
    #     latent_vf = self.mlp_extractor.forward_critic(features)
    #     return self.value_net(latent_vf)

class BaseMlpExtractor(nn.Module):
    def __init__(
            self,
            features_dim,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(BaseMlpExtractor, self).__init__()
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.latent_dim_pi = net_arch[-1] if len(net_arch) != 0 else features_dim
        self._build()

    def _build(self):
        self.mlp_extractor = create_mlp(
            self.features_dim,
            self.net_arch,
            activation_fn=self.activation_fn
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.mlp_extractor(obs)


class BaseApgPolicy(nn.Module):
    MlpExtractorAliases: ClassVar[Dict[str, Type[BaseMlpExtractor]]] = {"MlpExtractor": BaseMlpExtractor}
    FeaturesExtractorAliases: ClassVar[Dict[str, Type[BaseFeaturesExtractor]]] = \
        {"StateExtractor": StateExtractor, "StateTargetImageExtractor": StateTargetImageExtractor,
         "StateTargetExtractor": StateTargetExtractor}

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Optional[Union[str, Dict[str, Any]]],
            features_extractor_class: Optional[Type[BaseFeaturesExtractor]],
            mlp_extractor_class: Optional[Type[BaseMlpExtractor]],
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            mlp_extractor_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(BaseApgPolicy, self).__init__()
        # self.observation_space = observation_space
        # self.action_space = action_space
        self.features_extractor = self._build_features_extractor(
            observation_space=observation_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs
        )
        self.mlp_extractor = self._build_mlp_extractor(
            mlp_extractor_class=mlp_extractor_class,
            mlp_extractor_kwargs=mlp_extractor_kwargs
        )
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])
        # self.action_net = create_mlp(self.mlp_extractor.latent_dim_pi, [action_space.shape[0]], squash_output=True)
        self.lr_schedule = lr_schedule

    def _build_features_extractor(
            self,
            observation_space: spaces.Space,
            features_extractor_class: Type[BaseFeaturesExtractor],
            features_extractor_kwargs: Optional[Dict[str, Any]],
    ) -> Type[BaseFeaturesExtractor]:
        if features_extractor_class in self.FeaturesExtractorAliases:
            features_extractor_class = self.FeaturesExtractorAliases[features_extractor_class]
        else:
            features_extractor_class = features_extractor_class
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                "activation_fn": nn.ReLU,
                "net_arch": [],
            }
        features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)

        return features_extractor

    def _build_mlp_extractor(
            self,
            mlp_extractor_class: Type[BaseMlpExtractor],
            mlp_extractor_kwargs: Optional[Dict[str, Any]],
    ):
        if mlp_extractor_class in self.MlpExtractorAliases:
            mlp_extractor_class = self.MlpExtractorAliases[mlp_extractor_class]
        else:
            mlp_extractor_class = mlp_extractor_class
        if mlp_extractor_kwargs is None:
            mlp_extractor_kwargs = {
                "net_arch": [128, 128, 128],
                "activation_fn": nn.ReLU,
            }
        mlp_extractor = mlp_extractor_class(
            features_dim=self.features_extractor.features_dim,
            **mlp_extractor_kwargs
        )
        return mlp_extractor

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.action_net(self.mlp_extractor(self.features_extractor(obs)))

    def load_from_ppo(self):
        raise NotImplementedError


class StateApgPolicy(BaseApgPolicy):
    def __init__(self, observation_space, action_space, fea):
        pass
