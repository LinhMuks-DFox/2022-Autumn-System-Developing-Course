from ._FullyConnectedBasic import _FullyConnectedDense
from ._FullyConnectedClassifier import FullConnectedClassifier
from ._FullyConnectedRegressor import FullyConnectedRegressor

FullyConnectedDense = _FullyConnectedDense

__all__ = [
    "FullConnectedClassifier",
    "FullyConnectedRegressor",
    "FullyConnectedDense"
]
