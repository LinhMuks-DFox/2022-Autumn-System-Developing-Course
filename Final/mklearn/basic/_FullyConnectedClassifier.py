from typing import *

import torch
import torch.nn as nn

from mklearn.core.mk_types import *
from ._FullyConnectedBasic import _FullyConnectedDense


class FullConnectedClassifier(_FullyConnectedDense):

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_layer_cnt: Optional[int] = None,
                 hidden_layer_size: Optional[NDArray] = None,
                 device: Union[str, torch.device] = "cpu",
                 activation_type: Optional[Type[nn.Module]] = None):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_layer_cnt=hidden_layer_cnt,
            hidden_layer_size=hidden_layer_size,
            device=device,
            activation_type=activation_type
        )

    def summary(self) -> str:
        return "NeuralClassifier:" + str(self.properties_dict())

    def predict(self, x_sample):
        return torch.argmax(self.forward(x_sample))
