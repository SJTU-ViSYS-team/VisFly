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
from torch import Tensor
from torchvision import models


class CustomBaseFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Dict,
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        self._features_dim = 1
        super(CustomBaseFeaturesExtractor, self).__init__(observation_space, self._features_dim)
        self._is_recurrent = False
        self._extract_names = []
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

    @property
    def is_recurrent(self):
        return self._is_recurrent


def _get_conv_output(net, shape):
    net.eval()
    image = th.rand(1, *shape)
    output = net(image)
    net.train()
    return output.numel()


def _get_linear_output(net, shape):
    net.eval()
    image = th.rand(1, *shape)
    output = net(image)
    net.train()
    return output.numel()


def calc_required_input_dim(net, target_output_shape):
    """
    Calculate required input dimensions for a network with only transposed convolutions.

    Args:
        net: nn.Module containing only transposed convolution layers
        target_output_shape: tuple (H, W, C) desired output shape

    Returns:
        tuple: (H, W, C) required input shape
    """
    # Convert target shape to (C, H, W) as PyTorch uses channels first
    C, H, W = target_output_shape[0], target_output_shape[1], target_output_shape[2]

    # Extract all Conv2dTranspose layers
    trans_conv_layers = []
    for module in net.modules():
        if isinstance(module, nn.ConvTranspose2d):
            trans_conv_layers.append(module)

    if not trans_conv_layers:
        raise ValueError("Network must contain at least one ConvTranspose2d layer")

    # Work backwards through the layers
    curr_h, curr_w = H, W
    curr_c = C

    for layer in reversed(trans_conv_layers):
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size

        # Calculate required input dimensions
        curr_h = (curr_h + 2 * padding - kernel_size) // stride + 1
        curr_w = (curr_w + 2 * padding - kernel_size) // stride + 1
        curr_c = layer.in_channels

    # Return HWC format
    return (curr_c, curr_h, curr_w,)


class AutoTransDimBatchNorm1d(nn.BatchNorm1d):
    def __init__(self,*args, **kwargs):
        super(AutoTransDimBatchNorm1d, self).__init__(*args,**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.permute(1,2,0)
            output = super().forward(x)
            return output.permute(2,0,1)
        else:
            return super().forward(x)


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = th.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


def create_trans_cnn(
        kernel_size: List[int],
        channel: List[int],
        stride: List[int],
        padding: List[int],
        output_channel: Optional[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        bn: bool = False,
        ln: bool = False,
        bias: bool = True,
        target_output_shape: Tuple[int, int, int] = None,
        device: th.device = th.device("cpu")

) -> nn.Module:
    kernel_size = [kernel_size] * len(channel) if isinstance(kernel_size, int) else kernel_size
    stride = [stride] * len(channel) if isinstance(stride, int) else stride
    padding = [padding] * len(channel) if isinstance(padding, int) else padding

    assert len(kernel_size) == len(stride) == len(padding) == len(channel), \
        "The length of kernel_size, stride, padding and net_arch should be the same."

    modules = []

    prev_channel = channel[0]
    for idx in range(1, len(channel)):
        modules.append(nn.ConvTranspose2d(
            prev_channel,
            channel[idx],
            kernel_size=kernel_size[idx],
            stride=stride[idx],
            padding=padding[idx],
            bias=bias,
        )
        )
        prev_channel = channel[idx]
        if bn:
            modules.append(nn.BatchNorm2d(channel[idx]))
        if ln:
            modules.append(ImgChLayerNorm(channel[idx], eps=1e-3))
        modules.append(activation_fn())
        # modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if output_channel is not None:
        modules.append(nn.ConvTranspose2d(prev_channel, output_channel, kernel_size=kernel_size[-1], stride=stride[-1], padding=padding[-1]))

    if squash_output:
        modules.append(nn.Tanh())

    net = nn.Sequential(*modules).to(device)

    return net


def create_cnn(
        input_channels: int,
        kernel_size: List[int],
        channel: List[int],
        stride: List[int],
        padding: List[int],
        output_channel: Optional[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        bn: bool = False,
        ln: bool = False,
        bias: bool = True,
        max_pool: int = 0,
        device: th.device = th.device("cpu")

) -> nn.Module:
    kernel_size = [kernel_size] * len(channel) if isinstance(kernel_size, int) else kernel_size
    stride = [stride] * len(channel) if isinstance(stride, int) else stride
    padding = [padding] * len(channel) if isinstance(padding, int) else padding

    assert len(kernel_size) == len(stride) == len(padding) == len(channel), \
        "The length of kernel_size, stride, padding and net_arch should be the same."

    prev_channel = input_channels
    modules = []

    for idx in range(len(channel)):
        modules.append(nn.Conv2d(
            prev_channel,
            channel[idx],
            kernel_size=kernel_size[idx],
            stride=stride[idx],
            padding=padding[idx],
            bias=bias,
        )
        )
        prev_channel = channel[idx]
        if bn:
            modules.append(nn.BatchNorm2d(channel[idx]))
        if ln:
            modules.append(ImgChLayerNorm(channel[idx], eps=1e-3))
        modules.append(activation_fn())
        if max_pool > 0:
            modules.append(nn.MaxPool2d(kernel_size=max_pool, stride=2))

    if output_channel is not None:
        modules.append(nn.Conv2d(prev_channel, output_channel, kernel_size=kernel_size[-1], stride=stride[-1], padding=padding[-1]))

    modules.append(nn.Flatten())

    if squash_output:
        modules.append(nn.Tanh())

    net = nn.Sequential(*modules).to(device)
    return net


def create_mlp(
        input_dim: int,
        layer: List[int] = [],
        output_dim: Optional[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        bn: Union[bool, List] = False,
        squash_output: bool = False,
        bias: bool = True,
        ln: Union[bool, List] = False,
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
    :param bn: Whether to use batch normalization or not
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param ln: If set to False, Whether to use layer normalization or not
    :param device: Device on which the neural network should be run.

    :return:
    """
    # if batch_norm and layer_norm:
    #     raise ValueError("batch normalization and layer normalization should not be both implemented.")
    # bn = [bn] * len(layer) if isinstance(bn, bool) else bn
    # ln = [ln] * len(layer) if isinstance(ln, bool) else ln
    # for each_batch_norm, each_layer_norm in zip(bn, ln):
    #     assert not (each_batch_norm and each_layer_norm), "batch normalization and layer normalization should not be both implemented."

    # if input batch_norm list length is shorter than layer, then complete the list with False
    # if len(bn) < len(layer):
    #     bn += [False] * (len(layer) - len(bn))
    # if len(ln) < len(layer):
    #     ln += [False] * (len(layer) - len(ln))

    modules = []
    prev_dim = input_dim
    for idx in range(len(layer)):
        modules.append(nn.Linear(prev_dim, layer[idx], bias=bias))
        prev_dim = layer[idx]
        if bn:
            modules.append(AutoTransDimBatchNorm1d(layer[idx]))
        elif ln:
            modules.append(nn.LayerNorm(layer[idx], eps=1e-3))
        modules.append(activation_fn())

    if output_dim is not None:
        last_layer_dim = layer[-1] if len(layer) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))

    if squash_output:
        if len(modules) > 0 and not isinstance(modules[-1], nn.Linear):
            modules[-1] = nn.Tanh()
        else:
            modules.append(nn.Tanh())

    if len(modules) == 0:
        modules.append(nn.Flatten())

    net = nn.Sequential(*modules).to(device)

    output_dim = output_dim if output_dim is not None else prev_dim

    return net, output_dim


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
    layer = net_arch.get("layer", [])
    features_dim = layer[-1] if len(layer) != 0 else observation_space.shape[0]
    if hasattr(observation_space, "shape"):
        input_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape[1]
    else:
        input_dim = observation_space
    # input_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape[1]

    net = create_mlp(
        input_dim=input_dim,
        layer=net_arch.get("layer", []),
        activation_fn=activation_fn,
        bn=net_arch.get("bn", False),
        ln=net_arch.get("ln", False)
    )
    setattr(cls, name + "_extractor", net)
    cls._extract_names.append(name)

    features_dim = _get_linear_output(net, observation_space.shape)
    return features_dim


def set_trans_cnn_feature_extractor(cls, name, input_dim, target_shape, net_arch, activation_fn):
    net = create_trans_cnn(
        activation_fn=activation_fn,
        **net_arch,
    )
    required_input_dim = calc_required_input_dim(net, target_shape)
    flatten_dim = th.prod(th.as_tensor(required_input_dim))
    modules = [nn.Linear(input_dim, flatten_dim), net]
    cls._extract_names.append(name)
    setattr(cls, name + "_decoder", nn.Sequential(*modules))


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
            if net_arch.get("layer", None) is not None and len(net_arch.get("layer", [])) > 0:
                image_extractor.fc = create_mlp(
                    input_dim=image_extractor.fc.in_features,
                    activation_fn=activation_fn,
                    **net_arch
                )
        elif "efficientnet" in backbone:
            image_extractor.features[0][0] = nn.Conv2d(image_channels, image_extractor.features[0][0].out_channels,
                                                       kernel_size=image_extractor.features[0][0].kernel_size,
                                                       stride=image_extractor.features[0][0].stride,
                                                       padding=image_extractor.features[0][0].padding,
                                                       bias=image_extractor.features[0][0].bias is not None)

            if net_arch.get("layer", None) is not None and len(net_arch.get("layer", [])) > 0:
                image_extractor.classifier[-1] = create_mlp(
                    input_dim=image_extractor.classifier[-1].in_features,
                    activation_fn=activation_fn,
                    **net_arch
                )
        elif "mobilenet" in backbone:
            image_extractor.features[0][0] = nn.Conv2d(image_channels, image_extractor.features[0][0].out_channels,
                                                       kernel_size=image_extractor.features[0][0].kernel_size,
                                                       stride=image_extractor.features[0][0].stride,
                                                       padding=image_extractor.features[0][0].padding,
                                                       bias=image_extractor.features[0][0].bias is not None)
            if net_arch.get("layer", None) is not None and len(net_arch.get("layer", [])) > 0:
                image_extractor.classifier[-1] = create_mlp(
                    input_dim=image_extractor.classifier[-1].in_features,
                    activation_fn=activation_fn,
                    **net_arch
                )
        else:
            raise ValueError("Backbone not supported.")

    else:
        image_extractor = (
            create_cnn(
                input_channels=image_channels,
                kernel_size=net_arch.get("kernel_size", [5, 3, 3]),
                channel=net_arch.get("channel", [6, 12, 18]),
                activation_fn=activation_fn,
                padding=net_arch.get("padding", [0, 0, 0]),
                stride=net_arch.get("stride", [1, 1, 1]),
                bn=net_arch.get("bn", False),
                ln=net_arch.get("ln", False),
                bias=net_arch.get("bias", True),
            )
        )
        _image_features_dims = _get_conv_output(image_extractor, observation_space.shape)
        if len(net_arch.get("layer", [])) > 0:
            image_extractor.add_module("mlp",
                                       create_mlp(
                                           input_dim=_image_features_dims,
                                           layer=net_arch.get("layer"),
                                           activation_fn=activation_fn,
                                           bn=net_arch.get("bn", False),
                                           ln=net_arch.get("ln", False)
                                       )
                                       )
    setattr(cls, name + "_extractor", image_extractor)
    cls._extract_names.append(name)
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
        self._extract_names = []
        for key in observation_space.keys():
            if "semantic" in key or "color" in key or "depth" in key:
                _image_features_dims.append(
                    set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn)
                )
        self._features_dim = sum(_image_features_dims)

    def extract(self, observations) -> th.Tensor:
        features = []
        for name in self._extract_names:
            x = getattr(self, name + "_extractor")(observations[name])
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
        obs_keys = list(observation_space.keys())
        assert "state" in obs_keys
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


class SwarmStateTargetImageExtractor(StateTargetImageExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        obs_keys = list(observation_space.keys())
        assert "swarm" in obs_keys

        super(SwarmStateTargetImageExtractor, self).__init__(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn
        )

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn
        )
        _swarm_features_dim = set_mlp_feature_extractor(self, "swarm", observation_space["swarm"], net_arch.get("state", {}), activation_fn) * \
                              observation_space["swarm"].shape[0]
        self._features_dim = self._features_dim + _swarm_features_dim

    def extract(self, observation):
        swarm_features = []
        for i in range(observation["swarm"].shape[0]):
            swarm_features.append(self.swarm_extractor(observation["swarm"][i]))
        swarm_features = th.cat(swarm_features, dim=1)
        father_features = super().extract(observation)
        return th.cat([father_features, swarm_features], dim=1)


class StateGateExtractor(StateExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(StateGateExtractor, self).__init__(observation_space=observation_space, net_arch=net_arch, activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        gate_feature_dim = set_mlp_feature_extractor(self, "gate", observation_space["gate"], net_arch["gate"], activation_fn)
        self._features_dim = self._features_dim + gate_feature_dim

    def extract(self, observations) -> th.Tensor:
        state_features = self.state_extractor(observations['state'])
        gate_features = self.gate_extractor(observations['gate'])
        return th.cat([state_features, gate_features], dim=1)


class EmptyExtractor(CustomBaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(EmptyExtractor, self).__init__(observation_space=observation_space, net_arch=net_arch, activation_fn=activation_fn)
        self._features_dim = observation_space.shape[0]

    def _build(self, observation_space, net_arch, activation_fn):
        pass

    def extract(self, observations) -> th.Tensor:
        return observations


class LatentCombineExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(LatentCombineExtractor, self).__init__()
        _features_dim = 0
        for key in observation_space.keys():
            _features_dim += observation_space[key].shape[0]
        self._features_dim = _features_dim

    def _build(self, observation_space, net_arch, activation_fn):
        pass

    def extract(self, observations) -> th.Tensor:
        if isinstance(observations, dict):
            return th.cat([observations["stoch"], observations["deter"]], dim=-1)
        else:
            raise NotImplementedError

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, x):
        return self.extract(x)

class FlexibleMLP(nn.Module):
    extractor_alias: Dict = {
        "MLP": create_mlp,
        "CNN": create_cnn
    }

    def __init__(
            self,
            sub_extractors: List[BaseFeaturesExtractor],
            observation_space: spaces.Dict,
            net_arch: Dict,
            activation_fn: Type[nn.Module] = nn.ReLU,
            device: th.device = th.device("cpu"),
    ):
        super(FlexibleMLP, self).__init__(
            observation_space=observation_space,
        )
        self.device = device
        self._build(
            net_arch=net_arch,
            activation_fn=activation_fn,
            observation_space=observation_space,
            sub_extractors=sub_extractors
        )

    def _build(self, observation_space, net_arch, activation_fn, sub_extractors):
        def _build_sub_extractor(extractor_class, extractor_kwargs):
            if isinstance(extractor_class, str):
                extractor_func = self.extractor_alias["extractor_class"]
            else:
                raise NotImplementedError
            return extractor_func(input_dim=observation_space["extractor_class"].shape, **extractor_kwargs)

        features_dims = []
        for i, (ex_class, ex_kwargs) in enumerate(sub_extractors):
            features_dim = _build_sub_extractor(ex_class, ex_kwargs)
            features_dims.append(features_dim)

        self._features_dim = sum(features_dims)

        self.net = create_mlp(
            input_dim=self._features_dim,
            layer=net_arch.get("layer", []),
            activation_fn=activation_fn,
            bn=net_arch.get("bn", False),
            ln=net_arch.get("ln", False)
        )

    def forward(self, x):
        return self.net(x)
