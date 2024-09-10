from abc import abstractmethod
from distutils.dist import Distribution

import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from typing import Tuple, Callable, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from typing import List, Optional, Type, Union, Dict

from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from torchvision import models


class CustomBaseFeaturesExtractor(BaseFeaturesExtractor):
    is_recurrent = False

    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Dict,
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        self._features_dim = 1
        super(CustomBaseFeaturesExtractor, self).__init__(observation_space, self._features_dim)
        self._is_recurrent = False
        # self._build_recurrent()
        self._build(observation_space, net_arch, activation_fn)
        self._build_recurrent(net_arch)

    @abstractmethod
    def _build(self, observation_space, net_arch, activation_fn):
        pass

    def _build_recurrent(self, net_arch):
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    @abstractmethod
    def extract(self, observations) -> th.Tensor:
        pass

    def extract_with_recurrent(self, observations):
        features = self.extract(observations)
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(features.unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return features

    def forward(self, observations):
        return self.extract_with_recurrent(observations)


def _get_conv_output(net, shape):
    net.eval()
    image = th.rand(1, *shape)
    output = net(image)
    net.train()
    return output.numel()


def create_cnn(
        input_channels: int,
        kernel_size: List[int],
        channel: List[int],
        stride: List[int],
        padding: List[int],
        output_channel: Optional[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        batch_norm: bool = False,
        with_bias: bool = True,
        device: th.device = th.device("cpu")

) -> nn.Module:
    # assert len(kernel_size) == len(stride) == len(padding) == len(channel), \
    #     "The length of kernel_size, stride, padding and net_arch should be the same."

    if len(channel) > 0:
        modules = [nn.Conv2d(input_channels, channel[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])]
        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[0]))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
    else:
        modules = []

    for idx in range(len(channel) - 1):
        modules.append(nn.Conv2d(channel[idx], channel[idx + 1], kernel_size=kernel_size[idx + 1], stride=stride[idx + 1], padding=padding[idx + 1]))
        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[idx + 1]))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if output_channel is not None:
        last_layer_channel = channel[-1] if len(channel) > 0 else input_channels
        modules.append(nn.Conv2d(last_layer_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        if batch_norm:
            modules.append(nn.BatchNorm2d(output_channel))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

    modules.append(nn.Flatten())

    if squash_output:
        modules.append(nn.Tanh())

    net = nn.Sequential(*modules).to(device)
    return net


def create_mlp(
        input_dim: int,
        layer: List[int],
        output_dim: Optional[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        batch_norm: Union[bool, List] = False,
        squash_output: bool = False,
        with_bias: bool = True,
        layer_norm: Union[bool, List] = False,
        device: th.device = th.device("cpu")

) -> nn.Module:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param layer: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param batch_norm: Whether to use batch normalization or not
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param layer_norm: If set to False, Whether to use layer normalization or not
    :param device: Device on which the neural network should be run.

    :return:
    """
    # if batch_norm and layer_norm:
    #     raise ValueError("batch normalization and layer normalization should not be both implemented.")
    batch_norm = [batch_norm] * len(layer) if isinstance(batch_norm, bool) else batch_norm
    layer_norm = [layer_norm] * len(layer) if isinstance(layer_norm, bool) else layer_norm
    for each_batch_norm, each_layer_norm in zip(batch_norm, layer_norm):
        assert not (each_batch_norm and each_layer_norm), "batch normalization and layer normalization should not be both implemented."

    # if input batch_norm list length is shorter than layer, then complete the list with False
    if len(batch_norm) < len(layer):
        batch_norm += [False] * (len(layer) - len(batch_norm))
    if len(layer_norm) < len(layer):
        layer_norm += [False] * (len(layer) - len(layer_norm))

    if len(layer) > 0:
        modules = [nn.Linear(input_dim, layer[0], bias=with_bias)]
        if batch_norm[0]:
            modules.append(nn.BatchNorm1d(layer[0]))
        elif layer_norm[0]:
            modules.append(nn.LayerNorm(layer[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(layer) - 1):
        modules.append(nn.Linear(layer[idx], layer[idx + 1], bias=with_bias))
        if batch_norm[idx + 1]:
            modules.append(nn.BatchNorm1d(layer[idx + 1]))
        elif layer_norm[idx + 1]:
            modules.append(nn.LayerNorm(layer[idx + 1]))
        modules.append(activation_fn())

    if output_dim is not None:
        last_layer_dim = layer[-1] if len(layer) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
        # if batch_norm:
        #     modules.append(nn.BatchNorm1d(output_dim))
        # elif layer_norm:
        #     modules.append(nn.LayerNorm(output_dim))

    if squash_output:
        if len(modules) >= 0 and not isinstance(modules[-1], nn.Linear):
            modules[-1] = nn.Tanh()
        else:
            modules.append(nn.Tanh())

    if len(modules) == 0:
        modules.append(nn.Flatten())

    net = nn.Sequential(*modules).to(device)

    return net


def set_recurrent_feature_extractor(cls, input_size, rnn_setting):
    recurrent_alias = {
        "GRU": th.nn.GRU,
    }
    rnn_class = rnn_setting.get("class")
    kwargs = rnn_setting.get("kwargs")
    if isinstance(rnn_class, str):
        rnn_class = recurrent_alias[rnn_class]
    cls.__setattr__("recurrent_extractor", rnn_class(input_size=input_size, **kwargs))
    return kwargs.get("hidden_size")


def set_mlp_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    layer = net_arch.get("mlp_layer", [])
    features_dim = layer[-1] if len(layer) != 0 else observation_space.shape[0]
    input_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape[1]

    setattr(cls, name + "_extractor",
            create_mlp(
                input_dim=input_dim,
                layer=net_arch.get("mlp_layer", []),
                activation_fn=activation_fn,
                batch_norm=net_arch.get("bn", False),
                layer_norm=net_arch.get("ln", False)
            )
            )
    return features_dim


def set_cnn_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    image_channels = observation_space.shape[0]
    backbone = net_arch.get("backbone", None)
    if backbone is not None:
        image_extractor = cls.backbone_alias[backbone](pretrained=True)
        # replace the first layer to match the input channels
        if "resnet" in backbone:
            image_extractor.conv1 = nn.Conv2d(image_channels, image_extractor.conv1.out_channels,
                                              kernel_size=image_extractor.conv1.kernel_size,
                                              stride=image_extractor.conv1.stride,
                                              padding=image_extractor.conv1.padding,
                                              bias=image_extractor.conv1.bias is not None)
            if net_arch.get("mlp_layer", None) is not None and len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.fc = create_mlp(
                    input_dim=image_extractor.fc.in_features,
                    layer=net_arch.get("mlp_layer"),
                    activation_fn=activation_fn,
                    batch_norm=net_arch.get("bn", False),
                    layer_norm=net_arch.get("ln", False)
                )
        elif "efficientnet" in backbone:
            image_extractor.features[0][0] = nn.Conv2d(image_channels, image_extractor.features[0][0].out_channels,
                                                       kernel_size=image_extractor.features[0][0].kernel_size,
                                                       stride=image_extractor.features[0][0].stride,
                                                       padding=image_extractor.features[0][0].padding,
                                                       bias=image_extractor.features[0][0].bias is not None)

            if net_arch.get("mlp_layer", None) is not None and len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.classifier[-1] = create_mlp(
                    input_dim=image_extractor.classifier[-1].in_features,
                    layer=net_arch.get("mlp_layer"),
                    activation_fn=activation_fn,
                    batch_norm=net_arch.get("bn", False),
                    layer_norm=net_arch.get("ln", False)
                )
        elif "mobilenet" in backbone:
            image_extractor.features[0][0] = nn.Conv2d(image_channels, image_extractor.features[0][0].out_channels,
                                                       kernel_size=image_extractor.features[0][0].kernel_size,
                                                       stride=image_extractor.features[0][0].stride,
                                                       padding=image_extractor.features[0][0].padding,
                                                       bias=image_extractor.features[0][0].bias is not None)
            if net_arch.get("mlp_layer", None) is not None and len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.classifier[-1] = create_mlp(
                    input_dim=image_extractor.classifier[-1].in_features,
                    layer=net_arch.get("mlp_layer"),
                    activation_fn=activation_fn,
                    batch_norm=net_arch.get("bn", False),
                    layer_norm=net_arch.get("ln", False)
                )
        else:
            raise ValueError("Backbone not supported.")

    else:
        image_extractor = (
            create_cnn(
                input_channels=image_channels,
                kernel_size=net_arch.get("kernel_size", [5, 3, 3]),
                channel=net_arch.get("channels", [6, 12, 18]),
                activation_fn=activation_fn,
                padding=net_arch.get("padding", [0, 0, 0]),
                stride=net_arch.get("stride", [1, 1, 1]),
                batch_norm=net_arch.get("bn", False),

            )
        )
        _image_features_dims = _get_conv_output(image_extractor, observation_space.shape)
        if len(net_arch.get("mlp_layer", [])) > 0:
            image_extractor.add_module("mlp",
                                       create_mlp(
                                           input_dim=_image_features_dims,
                                           layer=net_arch.get("mlp_layer"),
                                           activation_fn=activation_fn,
                                           batch_norm=net_arch.get("bn", False),
                                           layer_norm=net_arch.get("ln", False)
                                       )
                                       )
    setattr(cls, name + "_extractor", image_extractor)
    cls._image_extractor_names.append(name + "_extractor")
    return _get_conv_output(image_extractor, observation_space.shape)


class TargetExtractor(CustomBaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Dict = {},
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "target" in observation_space.keys()
        super(TargetExtractor, self).__init__(observation_space=observation_space,
                                              net_arch=net_arch,
                                              activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        feature_dim = set_mlp_feature_extractor(
            self,
            name="target",
            observation_space=observation_space["target"],
            net_arch=net_arch["target"],
            activation_fn=activation_fn
        )
        self._features_dim = feature_dim

    def extract(self, observations) -> th.Tensor:
        return self.target_extractor(observations['target'])


class StateExtractor(CustomBaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Optional[Dict] = {},
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "state" in list(observation_space.keys())
        super(StateExtractor, self).__init__(observation_space=observation_space,
                                             net_arch=net_arch,
                                             activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        feature_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch["state"], activation_fn)
        self._features_dim = feature_dim

    def extract(self, observations) -> th.Tensor:
        return self.state_extractor(observations['state'])


class ImageExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # assume at least one image observation
        assert th.as_tensor([(key in "semantic") or ("color" in key) or ("depth" in key) for key in observation_space.keys()]).any()
        super(ImageExtractor, self).__init__(observation_space=observation_space,
                                             net_arch=net_arch,
                                             activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        # 处理image的卷积层
        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "semantic" in key or "color" in key or "depth" in key:
                _image_features_dims.append(
                    set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn)
                )
        self._features_dim = sum(_image_features_dims)

    def extract(self, observations) -> th.Tensor:
        features = []
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        combined_features = th.cat(features, dim=1)

        return combined_features


class StateTargetExtractor(CustomBaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) and ("target" in obs_keys)
        super(StateTargetExtractor, self).__init__(observation_space=observation_space,
                                                   net_arch=net_arch,
                                                   activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)

        self._features_dim = state_features_dim + target_features_dim

    def extract(self, observations):
        return th.cat([self.state_extractor(observations["state"]), self.target_extractor(observations["target"])], dim=1)


class StateImageExtractor(ImageExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        assert "state" in observation_space.keys()
        super(StateImageExtractor, self).__init__(observation_space=observation_space,
                                                  net_arch=net_arch,
                                                  activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        self._features_dim = _state_features_dim + self._features_dim

    def extract(self, observations) -> th.Tensor:
        state_features = self.state_extractor(observations['state'])
        features = [state_features]
        image_features = super().extract(observations)
        features.append(image_features)
        return th.cat(features, dim=1)


class StateTargetImageExtractor(ImageExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        assert ("target" in list(observation_space.keys())) and ("state" in list(observation_space.keys()))
        super(StateTargetImageExtractor, self).__init__(observation_space=observation_space,
                                                        net_arch=net_arch,
                                                        activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        self._features_dim = _state_features_dim + _target_features_dim + self._features_dim

    def extract(self, observation):
        state_features = self.state_extractor(observation['state'])
        target_features = self.target_extractor(observation['target'])
        image_features = super().extract(observation)
        return th.cat([state_features, target_features, image_features], dim=1)


class SwarmStateTargetImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys) \
               and ("swarm" in obs_keys)

        # 默认的特征维度为1
        self._features_dim = 1
        super(SwarmStateTargetImageExtractor, self).__init__(observation_space=observation_space,
                                                             features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _swarm_features_dim = set_mlp_feature_extractor(self, "swarm", observation_space["swarm"], net_arch.get("state", {}), activation_fn) * \
                              observation_space["swarm"].shape[0]

        # 处理image的卷积层
        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))

        self._features_dim = _state_features_dim + _target_features_dim + _swarm_features_dim + sum(_image_features_dims)

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        swarm_features = []
        for i in range(observations['swarm'].shape[1]):
            swarm_features.append(self.swarm_extractor(observations['swarm'][:, i, :]))
        # swarm_features = [self.state_extractor(agent) for agent in observations['swarm']]
        features = [state_features, target_features] + swarm_features
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # 合并state,target特征和image特征
        return th.cat(features, dim=1)


# class SwarmStateTargetImageExtractor(StateTargetImageExtractor):
#     backbone_alias: Dict = {
#         "resnet18": models.resnet18,
#         "resnet34": models.resnet34,
#         "resnet50": models.resnet50,
#         "resnet101": models.resnet101,
#         "efficientnet_l": models.efficientnet_v2_l,
#         "efficientnet_m": models.efficientnet_v2_m,
#         "efficientnet_s": models.efficientnet_v2_s,
#         "mobilenet_l": models.mobilenet_v3_large,
#         "mobilenet_s": models.mobilenet_v3_small,
#     }
#
#     def __init__(
#             self,
#             observation_space: spaces.Dict,
#             net_arch: Dict = {},
#             activation_fn: Type[nn.Module] = nn.ReLU,
#     ):
#         assert "swarm" in list(observation_space.keys())
#         super(SwarmStateTargetImageExtractor, self).__init__(observation_space=observation_space,
#                                                              net_arch=net_arch,
#                                                              activation_fn=activation_fn)
#
#     def _build(self, observation_space, net_arch, activation_fn):
#         super()._build(
#             observation_space=observation_space,
#             net_arch=net_arch,
#             activation_fn=activation_fn
#         )
#         _swarm_features_dim = set_mlp_feature_extractor(self, "swarm", observation_space["swarm"], net_arch.get("state", {}), activation_fn) * \
#                               observation_space["swarm"].shape[0]
#         self._features_dim += _swarm_features_dim
#
#     def extract(self, observations) -> th.Tensor:
#         super_features = super().extract(observations)
#         swarm_features = []
#         for i in range(observations['swarm'].shape[1]):
#             # each agent state observation as batch
#             swarm_features.append(self.swarm_extractor(observations['swarm'][:, i, :]))
#         return th.cat([super_features] + swarm_features, dim=1)


