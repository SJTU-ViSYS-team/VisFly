from dataclasses import dataclass
import torch as th
from typing import Union, Any
from enum import Enum
import numpy as np


@dataclass
class bound:
    min: float
    max: float


class ACTION_TYPE(Enum):
    THRUST = 0
    BODYRATE = 1
    VELOCITY = 2
    POSITION = 3


class Uniform:
    mean: Union[float, th.Tensor] = 0
    half: Union[float, th.Tensor] = 0

    def __init__(
            self,
            mean,
            half, ):
        self.mean = th.as_tensor(mean)
        self.half = th.as_tensor(half)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.half = self.half.to(device)
        return self

    def generate(self, size):
        return (th.rand(size) - 0.5) * self.half + self.mean


class Normal:
    mean: Union[float, th.Tensor] = 0
    std: Union[float, th.Tensor] = 0

    def __init__(
            self,
            mean,
            std, ):
        self.mean = th.as_tensor(mean)
        self.std = th.as_tensor(std)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def generate(self, size):
        return th.normal(self.mean, self.std, size)


@dataclass
class PID:
    p: th.Tensor = th.diag(th.tensor([1, 1, 1]))
    i: th.Tensor = th.diag(th.tensor([1, 1, 1]))
    d: th.Tensor = th.diag(th.tensor([1, 1, 1]))

    def to(self, device):
        self.p = self.p.to(device)
        self.i = self.i.to(device)
        self.d = self.d.to(device)
        return self

    def clone(self):
        self.p = self.p.clone()
        self.i = self.i.clone()
        self.d = self.d.clone()

        return self

    def detach(self):
        self.p = self.p.detach()
        self.i = self.i.detach()
        self.d = self.d.detach()

        return self


class SortDict(dict):
    def __init__(self, data):
        super().__init__(data)

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, str):
            return super().__getitem__(__key)
        else:
            # return super().__getitem__(list(self.keys())[__key])
            return dict([(key, super().__getitem__(key)[__key]) for key in self.keys()])


# create a dict designed for tensor that can use detach
class TensorDict(dict):
    def __init__(self, data):
        super().__init__(data)

    # return a new detach, do not change instance itself
    def detach(self):
        return TensorDict({key: self[key].detach() for key in self.keys()})

    def clone(self):
        for key in self.keys():
            self[key] = self[key].clone()

        return self

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, int):
            return TensorDict({k: v[key] for k, v in self.items()})
        elif hasattr(key, "__iter__"):
            return TensorDict({k: v[key] for k, v in self.items()})
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, (int, th.Tensor, np.ndarray, list)):
            for k in self.keys():
                self[k][key] = value[k]
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def append(self, data):
        if isinstance(data, TensorDict):
            for key, value in data.items():
                self[key] = th.cat([self[key], data[key]])

    def cpu(self):
        for key, value in self.items():
            self[key] = self[key].cpu()

    def as_tensor(self, device=th.device("cpu")):
        d = {}
        for key, value in self.items():
            d[key] = th.as_tensor(value, device=device)

        return d

    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self

