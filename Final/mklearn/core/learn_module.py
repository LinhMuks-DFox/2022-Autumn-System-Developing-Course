import abc
from typing import *

import torch.nn


class LearnModule(torch.nn.Module):

    @abc.abstractmethod
    def to_device(self, device: Union[str, torch.device]):
        pass

    @abc.abstractmethod
    def summary(self) -> str:
        pass

    @abc.abstractmethod
    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        pass

    def __str__(self):
        return self.summary()

    __repr__ = __str__

    def predict(self, x_sample):
        return self.forward(x_sample)
