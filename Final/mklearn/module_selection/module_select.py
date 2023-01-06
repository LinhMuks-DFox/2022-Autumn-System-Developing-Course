from typing import Tuple

import numpy as np

from mklearn.core.mk_types import *


class DataSetSplitter:
    def __init__(self, test_ratio: float = 0.2, seed: int = None):
        self.train_test_split_ratio_ = test_ratio
        assert 0.0 <= self.train_test_split_ratio_ <= 1.0
        if seed:
            np.random.seed(seed)

    def split(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        shuffled_idx = np.random.permutation(len(data))
        test_set_size = int(len(data) * self.train_test_split_ratio_)
        test_indices = shuffled_idx[:test_set_size]
        train_indices = shuffled_idx[test_set_size:]

        return data[train_indices], data[test_indices], label[train_indices], label[test_indices]

    def __call__(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        return self.split(data, label)

    def forward(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        return self.split(data, label)


def train_test_split(data: NDArray,
                     label: NDArray,
                     test_ratio: float = 0.2,
                     seed: int = None) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    splitter = DataSetSplitter(test_ratio, seed)
    return splitter(data, label)


def train_test_validate_split(data: NDArray,
                              label: NDArray,
                              test_ratio: float = 0.2,
                              validate_ratio: float = 0.2,
                              seed: int = None) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    splitter = DataSetSplitter(test_ratio, seed)
    train_data, test_data, train_label, test_label = splitter(data, label)

    splitter = DataSetSplitter(validate_ratio, seed)
    train_data, validate_data, train_label, validate_label = splitter(train_data, train_label)

    return train_data, validate_data, test_data, train_label, validate_label, test_label
