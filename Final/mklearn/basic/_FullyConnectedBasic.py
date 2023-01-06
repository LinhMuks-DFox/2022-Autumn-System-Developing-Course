from typing import Dict, Any, Union, Optional, Type, Iterator

import numpy as np
import torch
import torch.nn as nn

from mklearn.core.learn_module import LearnModule
from mklearn.core.mk_types import *
from mklearn.core.mklearn_errors import ShapeError


class _FullyConnectedDense(LearnModule):
    __slots__ = [
        "activation_fn_type_",
        "device_",
        "hidden_layer_cnt_",
        "hidden_layer_size_",
        "input_shape_",
        "layer_lists_",
        "layer_seq_",
        "output_shape_"
    ]

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_layer_cnt: Optional[int] = None,
                 hidden_layer_size: Optional[NDArray] = None,
                 device: Union[str, torch.device] = "cpu",
                 activation_type: Optional[Type[nn.Module]] = None):
        super().__init__()
        # parameter check
        self.input_shape_: int = input_shape
        self.output_shape_: int = output_shape
        self.hidden_layer_cnt_: int = hidden_layer_cnt if hidden_layer_cnt is not None else 3
        self.hidden_layer_size_: NDArray = hidden_layer_size if hidden_layer_size is not None else np.array(
            [input_shape // 2, input_shape // 4, self.output_shape_])
        self.activation_fn_type_: Type[nn.Module] = activation_type if activation_type is not None else nn.ReLU
        if self.hidden_layer_cnt_ != len(self.hidden_layer_size_):
            raise ShapeError("hidden layer count not equal to hidden_layer_size_")
        if len(self.hidden_layer_size_.shape) != 1:
            raise ShapeError(f"hidden layer count not equal to 1, but{len(self.hidden_layer_size_)}")
        self.device_ = device

        # build layer sequence
        self.layer_lists_ = [nn.Linear(in_features=self.input_shape_, out_features=self.hidden_layer_size_[0]),
                             self.activation_fn_type_()]

        # build hidden layers
        for i in range(1, self.hidden_layer_cnt_ - 1):
            self.layer_lists_.append(
                nn.Linear(in_features=self.hidden_layer_size_[i - 1], out_features=self.hidden_layer_size_[i]))
            self.layer_lists_.append(self.activation_fn_type_())
        # build output layer
        self.layer_lists_.append(nn.Linear(in_features=self.hidden_layer_size_[-2], out_features=self.output_shape_))

        # convert to sequential
        self.layer_seq_: Optional[nn.Sequential] = nn.Sequential(*self.layer_lists_)

        self.to_device(self.device_)

    def to_device(self, device: Union[str, torch.device]):
        if self.layer_seq_ is None:
            raise AttributeError("Model was not initialized correctly")
        self.device_ = device
        self.layer_seq_.to(self.device_ if isinstance(self.device_, torch.device) else torch.device(self.device_))

    def summary(self) -> str:
        return "_FullyConnectedDense:" + str(self.properties_dict())

    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        return {
            "input_shape": self.input_shape_,
            "output_shape": self.output_shape_,
            "hidden_layer_cnt": self.hidden_layer_cnt_,
            "hidden_layer_size": self.hidden_layer_size_,
            "activation_type": self.activation_fn_type_,
            "device": self.device_
        }

    def forward(self, x_sample):
        return self.layer_seq_(x_sample)

    def predict(self, x_sample):
        return self.forward(x_sample)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.layer_seq_.parameters(recurse=recurse)
