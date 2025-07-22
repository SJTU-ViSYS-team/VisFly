from abc import abstractmethod

import gym
import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable, Any
from gym import spaces
from typing import List, Optional, Type, Union, Dict

from torch import Tensor
from torchvision import models


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


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

    def extract(self, observations) -> th.Tensor:
        features = []
        for name in self._extract_names:
            is_exceed_dim = (len(observations[name].shape) > 4) and \
                name in ["semantic", "color", "depth"]
            if is_exceed_dim:
                obs = observations[name].reshape(-1, *observations[name].shape[-3:])
            else:
                obs = observations[name]
            x = getattr(self, name + "_extractor")(obs)
            if is_exceed_dim:
                x = x.reshape(*observations[name].shape[:-3], x.shape[-1])
            features.append(x)
        combined_features = th.cat(features, dim=-1)

        return combined_features

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


def compute_required_input_shape(
    net: nn.Module,
    desired_out: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    反向遍历网络，逐层还原所需的输入 (C, H, W)。
    """

    def _invert_size(o: int, k: int, s: int, p: int,
                     d: int, op: int) -> int:
        """
        单层 ConvTranspose2d 的逆向尺寸公式：
            o = (i - 1) * s - 2p + d*(k - 1) + op + 1
        反推：
            i = (o - 1 - d*(k - 1) - op + 2p) / s + 1
        """
        num = o - 1 - d * (k - 1) - op + 2 * p
        if num % s:
            raise ValueError(
                f'无法整除：({o}-1-{d}*({k}-1)-{op}+2*{p}) 不能被 stride={s} 整除')
        return num // s + 1

    C_out, H, W = desired_out
    # 仅保留 ConvTranspose2d 层，保持拓扑顺序
    deconv_layers = [
        m for m in net.modules() if isinstance(m, nn.ConvTranspose2d)
    ]

    if not deconv_layers:
        raise ValueError("网络中未找到 nn.ConvTranspose2d 层")

    # 验证最后一层输出通道
    last_layer = deconv_layers[-1]
    if last_layer.out_channels != C_out:
        raise ValueError(
            f"期望输出通道 {C_out} 与网络最后一层 "
            f"out_channels={last_layer.out_channels} 不一致"
        )

    # 逆序推回输入空间尺寸
    for layer in reversed(deconv_layers):
        H = _invert_size(
            H,
            k=layer.kernel_size[0],
            s=layer.stride[0],
            p=layer.padding[0],
            d=layer.dilation[0],
            op=layer.output_padding[0]
        )
        W = _invert_size(
            W,
            k=layer.kernel_size[1],
            s=layer.stride[1],
            p=layer.padding[1],
            d=layer.dilation[1],
            op=layer.output_padding[1]
        )

    C_in = deconv_layers[0].in_channels  # 第一层（逆向后）输入通道
    return (C_in, H, W)


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


class SafeFlatten(nn.Module):
    """A safe flatten layer that uses keyword arguments to avoid PyTorch compatibility issues."""
    def __init__(self, start_dim=1, end_dim=-1):
        super(SafeFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x):
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


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

    modules.append(SafeFlatten())

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
        modules.append(nn.Identity())

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

    net, output_dim = create_mlp(
        input_dim=input_dim,
        layer=layer,
        activation_fn=activation_fn,
        bn=net_arch.get("bn", False),
        ln=net_arch.get("ln", False)
    )
    setattr(cls, name + "_extractor", net)
    if not hasattr(cls, '_extract_names'):
        cls._extract_names = []
    cls._extract_names.append(name)

    # features_dim = _get_linear_output(net, observation_space.shape)
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
    image_channels = observation_space.shape[0]
    backbone = net_arch.get("backbone", None)
    modules = []
    if backbone is not None:
        pre_process_layer = nn.Conv2d(image_channels, 3,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=True)
        modules.append(pre_process_layer)
        image_extractor = backbone_alias[backbone](pretrained=True)
    else:
        image_extractor = create_cnn(
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
    modules.append(image_extractor)
    cache_net = nn.Sequential(*modules)
    conv_output = _get_conv_output(cache_net, observation_space.shape)
    aft_process_layer, _output_dim = create_mlp(
                                       input_dim=conv_output,
                                       layer=net_arch.get("layer",[]),
                                       activation_fn=activation_fn,
                                       bn=net_arch.get("bn", False),
                                       ln=net_arch.get("ln", False)
                                   )
    modules.append(aft_process_layer)

    net = nn.Sequential(*modules)
    setattr(cls, name + "_extractor", net)
    cls._extract_names.append(name)
    return _output_dim


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


class FlexibleExtractor(CustomBaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        super(FlexibleExtractor, self).__init__(observation_space=observation_space,
                                                 net_arch=net_arch,
                                                 activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        self._features_dim = 0
        for key, value in net_arch.items():
            if "semantic" in key or "color" in key or "depth" in key:
                _image_features_dim = set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn)
                self._features_dim += _image_features_dim
            elif "state" in key:
                _state_features_dim = set_mlp_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn)
                self._features_dim += _state_features_dim


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
        _image_features_dim = self._features_dim  # Store the image features dimension from parent
        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        
        # Check if there are actually any image features (extractors with image-like names)
        has_image_features = any("semantic" in name or "color" in name or "depth" in name 
                                for name in self._extract_names)
        
        if has_image_features:
            self._features_dim = _state_features_dim + _target_features_dim + _image_features_dim
        else:
            self._features_dim = _state_features_dim + _target_features_dim


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


class StateGateExtractor(StateExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(StateGateExtractor, self).__init__(observation_space=observation_space, net_arch=net_arch, activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        gate_feature_dim = set_mlp_feature_extractor(self, "gate", observation_space["gate"], net_arch["gate"], activation_fn)
        self._features_dim = self._features_dim + gate_feature_dim


class EmptyExtractor(CustomBaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(EmptyExtractor, self).__init__(observation_space=observation_space, net_arch=net_arch, activation_fn=activation_fn)
        # self._features_dim = observation_space.shape[0]

    def _build(self, observation_space, net_arch, activation_fn):
        pass


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

        self.net, output_dim = create_mlp(
            input_dim=self._features_dim,
            layer=net_arch.get("layer", []),
            activation_fn=activation_fn,
            bn=net_arch.get("bn", False),
            ln=net_arch.get("ln", False)
        )

    def forward(self, x):
        return self.net(x)


def load_extractor_class(cls):
    cls_alias = {
        "StateExtractor": StateExtractor,
        "ImageExtractor": ImageExtractor,
        "TargetExtractor": TargetExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
        "SwarmStateTargetImageExtractor": SwarmStateTargetImageExtractor,
        "StateGateExtractor": StateGateExtractor,
        "EmptyExtractor": EmptyExtractor,
        "LatentCombineExtractor": LatentCombineExtractor,
        "FlexibleExtractor": FlexibleExtractor,
    }
    if cls in cls_alias.keys():
        return cls_alias[cls]
    else:
        raise ValueError(f"Extractor class {cls} not found in alias. Available classes: {list(cls_alias.keys())}")

