import typing

import numpy as np
import torch

NDArray = typing.TypeVar('NDArray', bound=np.ndarray)
NDTensor = typing.TypeVar("NDTensor", bound=torch.Tensor)

__all__ = [
    "NDArray",
    "NDTensor"
]
