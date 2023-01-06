import copy
from typing import *

import numpy as np
import torch.nn as nn
import torch.utils.data

from mklearn.core.learn_module import LearnModule
from mklearn.core.mk_types import *


class ConvNDimNTimes(LearnModule):
    __slots__ = [
        "active_function_type_",
        "conv_bias_",
        "conv_dilation_",
        "conv_groups_",
        "conv_kernel_size_",
        "conv_layer_output_channels_",
        "conv_layer_type_",
        "conv_layers_lists_",
        "conv_n_times_",
        "conv_padding_",
        "conv_padding_mode_",
        "conv_seq_",
        "conv_stride_",
        "device_",
        "first_conv_layer_in_channel_",
        "input_dim_",
        "receptive_field_",
        "shape_after_convolution_"
    ]

    def __init__(self,
                 input_dim: NDArray,
                 conv_n_times: int,
                 kernel_sizes: Optional[NDArray] = None,
                 out_channels: Optional[NDArray] = None,
                 paddings: Optional[NDArray] = None,
                 strides: Optional[NDArray] = None,
                 dilation: Optional[NDArray] = None,
                 conv_layer_type: Optional[Union[Type[nn.Conv1d], Type[nn.Conv2d], Type[nn.Conv3d]]] = None,
                 groups: Optional[List[int]] = None,
                 bias: Optional[List[bool]] = None,
                 active_function: Optional[Union[Type[nn.ReLU], Type[nn.Sigmoid], Type[nn.Softmax]]] = None,
                 padding_mode: Optional[List[str]] = None,
                 device: Optional[Union[str, torch.device]] = "cpu",
                 ):
        """
        Create n_time n_dim convolutional layers. Can be used in nn.Sequential.
        :param input_dim: Eg: [Channel, Height, Width], 3d: [Channel, Depth, Height, Width], No Batch Size
        :param conv_n_times: How many times to do convolution
        :param kernel_sizes: Eg: [3, 3], 3d: [3, 3, 3]
        :param out_channels: Eg: [16, 32], 3d: [16, 32]
        :param paddings: Eg: [1, 1], 3d: [1, 1, 1]
        :param strides: Eg: [1, 1], 3d: [1, 1, 1]
        :param dilation: Eg: [1, 1], 3d: [1, 1, 1]
        :param conv_layer_type: Eg: nn.Conv2d
        :param groups: Eg: [1, 1]
        :param bias: Eg: [True, True]
        :param active_function: active function between conv layers
        :param padding_mode: Eg: ["zeros", "zeros"]
        :param device: Eg: "cpu"
        """
        super().__init__()

        # parameter check
        self.input_dim_: NDArray = input_dim
        self.conv_n_times_: int = conv_n_times
        self.first_conv_layer_in_channel_: int = self.input_dim_[0]
        self.conv_kernel_size_: NDArray = kernel_sizes \
            if kernel_sizes is not None else np.array([[3, 3] * self.conv_n_times_])
        self.conv_layer_output_channels_: NDArray = out_channels \
            if out_channels is not None else np.array([1 for _ in range(self.conv_n_times_)])

        self.conv_padding_: NDArray = paddings \
            if paddings is not None else np.array([[0, 0] for _ in range(self.conv_n_times_)])
        self.conv_stride_: NDArray = strides \
            if strides is not None else np.array([[1, 1] for _ in range(self.conv_n_times_)])
        self.conv_dilation_: NDArray = dilation \
            if dilation is not None else np.array([[1, 1] for _ in range(self.conv_n_times_)])
        self.conv_layer_type_ = conv_layer_type \
            if conv_layer_type is not None else \
            {
                1: nn.Conv1d,
                2: nn.Conv2d,
                3: nn.Conv3d
            }.get(len(input_dim) - 1)
        self.conv_groups_ = groups \
            if groups is not None else [1 for _ in range(self.conv_n_times_)]

        self.conv_bias_ = bias \
            if bias is not None else [False for _ in range(self.conv_n_times_)]

        self.active_function_type_ = active_function \
            if active_function is not None else nn.ReLU
        self.conv_padding_mode_: List[str] = padding_mode \
            if padding_mode is not None else ["zeros" for _ in range(self.conv_n_times_)]

        if not len(self.conv_kernel_size_) == self.conv_n_times_ == \
               len(self.conv_layer_output_channels_) == len(self.conv_padding_mode_):
            raise ValueError(
                f"size of kernel_size({len(self.conv_kernel_size_)}) and conv_n_times({self.conv_n_times_}) "
                f"and sizeof conv_layer_output_channels({len(self.conv_layer_output_channels_)}) and "
                f"padding mode length{len(self.conv_padding_mode_)} shall be equal.")

        # compute convolution properties
        self.shape_after_convolution_ = ConvNDimNTimes.shape_after_n_time_convolution(
            ndim_shape=self.input_dim_[1:],
            n_time=self.conv_n_times_,
            kernel_size=self.conv_kernel_size_,
            padding=self.conv_padding_,
            stride=self.conv_stride_,
            dilation=self.conv_dilation_
        )

        self.receptive_field_ = ConvNDimNTimes.receptive_field(self.conv_kernel_size_)

        # build convolution input layers
        self.conv_layers_lists_ = [
            self.conv_layer_type_(in_channels=self.first_conv_layer_in_channel_,
                                  out_channels=self.conv_layer_output_channels_[0],
                                  kernel_size=self.conv_kernel_size_[0],
                                  padding=self.conv_padding_[0],
                                  stride=self.conv_stride_[0],
                                  dilation=self.conv_dilation_[0],
                                  groups=self.conv_groups_[0],
                                  bias=self.conv_bias_[0],
                                  padding_mode=self.conv_padding_mode_[0]
                                  ),
            self.active_function_type_()
        ]
        # build convolution hidden(conv) layers
        for i in range(1, self.conv_n_times_):
            self.conv_layers_lists_.append(
                self.conv_layer_type_(in_channels=self.conv_layer_output_channels_[i - 1],
                                      out_channels=self.conv_layer_output_channels_[i],
                                      kernel_size=self.conv_kernel_size_[i],
                                      padding=self.conv_padding_[i],
                                      stride=self.conv_stride_[i],
                                      dilation=self.conv_dilation_[i],
                                      groups=self.conv_groups_[i],
                                      bias=self.conv_bias_[i],
                                      padding_mode=self.conv_padding_mode_[i]))
            # No active function after last conv layer
            if i < self.conv_n_times_ - 1:
                self.conv_layers_lists_.append(self.active_function_type_())

        self.device_ = torch.device(device)

        self.conv_seq_ = nn.Sequential(*self.conv_layers_lists_)

        self.conv_seq_.to(self.device_)

    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim_.copy(),
            "conv_n_times": self.conv_n_times_,
            "first_conv_layer_in_channel": self.first_conv_layer_in_channel_,
            "conv_kernel_size": self.conv_kernel_size_.copy(),
            "conv_layer_output_channels": self.conv_layer_output_channels_.copy(),
            "conv_padding": self.conv_padding_.copy(),
            "conv_stride": self.conv_stride_.copy(),
            "conv_dilation": self.conv_dilation_.copy(),
            "conv_layer_type": self.conv_layer_type_,
            "conv_groups": self.conv_groups_.copy(),
            "conv_bias": self.conv_bias_.copy(),
            "active_function_type": self.active_function_type_,
            "conv_padding_mode": self.conv_padding_mode_.copy(),
            "shape_after_convolution": self.shape_after_convolution_.copy(),
            "receptive_field": self.receptive_field_.copy(),
            "conv_layers_lists": (layers := copy.deepcopy(self.conv_layers_lists_)),
            "device": self.device_,
            "conv_seq": (nn.Sequential(*layers)).to(self.device_)
        }

    def forward(self, x_sample: torch.Tensor):
        return self.conv_seq_(x_sample)

    predict = forward

    def to_device(self, device: Union[str, torch.device]):
        self.device_ = device if isinstance(device, torch.device) else torch.device(device)
        self.conv_seq_.to(self.device_)
        return self

    @staticmethod
    def shape_after_conv(ndim_shape: NDArray,
                         kernel_size: NDArray,
                         padding: NDArray = (0,),
                         stride: NDArray = (1,),
                         dilation: NDArray = (1,)) -> NDArray:
        """
        Example:
        For a date in 2 dimension:
            ndim_shape:[28, 28],
        kernel size shall be an array, even the kernel is a Square, shape=(3, 3), as well as padding, string, dilation
        :param ndim_shape:
        :param kernel_size:
        :param padding:
        :param stride:
        :param dilation:
        :return:
        """
        if not np.all(isinstance(i, np.ndarray) for i in [ndim_shape, kernel_size, padding, stride, dilation]):
            raise TypeError(f"Function args shall be a instance of np.ndarray, but:\n"
                            f"ndim_shape is {type(ndim_shape)}, "
                            f"kernel_size is {type(kernel_size)}\n"
                            f"padding is {type(padding)}, "
                            f"stride is {type(stride)}\n"
                            f"dilation is {type(dilation)}\n")
        return ((ndim_shape + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1

    @staticmethod
    def shape_after_conv_transpose(
            ndim_shape: NDArray,
            kernel_size: NDArray,
            padding: NDArray = (0,),
            stride: NDArray = (1,),
            dilation: NDArray = (1,), ) -> NDArray:
        if not np.all(isinstance(i, np.ndarray) for i in [ndim_shape, kernel_size, padding, stride, dilation]):
            raise TypeError(f"Function args shall be a instance of np.ndarray, but:\n"
                            f"ndim_shape is {type(ndim_shape)}, "
                            f"kernel_size is {type(kernel_size)}\n"
                            f"padding is {type(padding)}, "
                            f"stride is {type(stride)}\n"
                            f"dilation is {type(dilation)}\n")
        return (ndim_shape - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1

    @staticmethod
    def shape_after_n_time_convolution(
            ndim_shape: NDArray,
            kernel_size: NDArray,
            n_time: int = 1,
            padding: NDArray = (0,),
            stride: NDArray = (1,),
            dilation: NDArray = (1,),

    ) -> NDArray:
        ret = ndim_shape
        for i in range(n_time):
            ret = ConvNDimNTimes.shape_after_conv(
                ret,
                kernel_size=kernel_size[i],
                padding=padding[i],
                stride=stride[i],
                dilation=dilation[i]
            )
        return ret

    @staticmethod
    def shape_after_n_time_convolution_transpose(
            ndim_shape: NDArray,
            kernel_size: NDArray,
            n_time: int = 1,
            padding: NDArray = (0,),
            stride: NDArray = (1,),
            dilation: NDArray = (1,),

    ) -> NDArray:
        ret = ndim_shape
        for i in range(n_time):
            ret = ConvNDimNTimes.shape_after_conv_transpose(
                ret,
                kernel_size=kernel_size[i],
                padding=padding[i],
                stride=stride[i],
                dilation=dilation[i]
            )
        return ret

    @staticmethod
    def receptive_field(kernel_size: NDArray) -> NDArray:
        return np.sum(kernel_size, axis=0) - 1

    def summary(self) -> str:
        return "ConvNDimNTimes(\n" + str(self.properties_dict()) + "\n)"

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.conv_seq_.parameters(recurse=recurse)
