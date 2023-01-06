import copy
from typing import *

import numpy as np
import torch.nn as nn
import torch.utils.data

from mklearn.core.learn_module import LearnModule
from mklearn.core.mk_types import *


class ConvTransposeNDimNTimes(LearnModule):
    __slots__ = [
        "active_function_type_",
        "conv_bias_",
        "conv_dilation_",
        "conv_first_layer_in_channel_",
        "conv_groups_",
        "conv_kernel_size_",
        "conv_output_channels_",
        "conv_padding_",
        "conv_padding_modes_",
        "conv_stride_",
        "conv_transpose_n_times_",
        "conv_transpose_seq_",
        "convolution_transpose_layer_lists_",
        "convolution_transpose_layer_type_",
        "device_",
    ]

    def __init__(self,
                 input_dim: NDArray,
                 conv_transpose_n_times: int,
                 kernel_size: Optional[NDArray] = None,
                 out_channels: Optional[NDArray] = None,
                 padding: Optional[NDArray] = None,
                 stride: Optional[NDArray] = None,
                 dilation: Optional[NDArray] = None,
                 convolution_transpose_layer_type:
                 Optional[Union[Type[nn.ConvTranspose1d], Type[nn.ConvTranspose2d], Type[nn.ConvTranspose3d]]] = None,
                 groups: Optional[List[int]] = None,
                 bias: Optional[List[bool]] = None,
                 active_function: Optional[Union[Type[nn.ReLU], Type[nn.Sigmoid], Type[nn.Softmax]]] = None,
                 padding_modes: Optional[List[str]] = None,
                 device: Optional[Union[str, torch.device]] = "cpu",
                 ):
        super(ConvTransposeNDimNTimes, self).__init__()
        # parameters check
        self.conv_first_layer_in_channel_ = input_dim[0]
        self.conv_transpose_n_times_: int = conv_transpose_n_times
        self.conv_kernel_size_: NDArray = kernel_size \
            if kernel_size is not None else \
            np.array([[3, 3] for _ in range(self.conv_transpose_n_times_)])
        self.conv_output_channels_: NDArray = out_channels \
            if out_channels is not None else \
            np.array([1 for _ in range(self.conv_transpose_n_times_)])
        self.conv_padding_: NDArray = padding \
            if padding is not None else \
            np.array([[0, 0] for _ in range(self.conv_transpose_n_times_)])
        self.conv_stride_: NDArray = stride \
            if stride is not None else \
            np.array([[1, 1] for _ in range(self.conv_transpose_n_times_)])
        self.conv_dilation_: NDArray = dilation \
            if dilation is not None else \
            np.array([[1, 1] for _ in range(self.conv_transpose_n_times_)])
        self.convolution_transpose_layer_type_ = convolution_transpose_layer_type \
            if convolution_transpose_layer_type is not None else \
            {
                1: nn.ConvTranspose1d,
                2: nn.ConvTranspose2d,
                3: nn.ConvTranspose3d
            }.get(len(input_dim) - 1)
        self.conv_groups_: List[int] = groups \
            if groups is not None else \
            [1 for _ in range(self.conv_transpose_n_times_)]
        self.conv_bias_: List[bool] = bias \
            if bias is not None else \
            [False for _ in range(self.conv_transpose_n_times_)]
        self.conv_padding_modes_: List[str] = padding_modes \
            if padding_modes is not None else \
            ["zeros" for _ in range(self.conv_transpose_n_times_)]
        self.active_function_type_ = active_function \
            if active_function is not None else nn.ReLU
        self.device_ = device

        self.shape_after_n_time_convolution_transpose_ = self.shape_after_n_time_convolution_transpose(
            ndim_shape=input_dim[1:],
            n_time=self.conv_transpose_n_times_,
            kernel_size=self.conv_kernel_size_,
            padding=self.conv_padding_,
            stride=self.conv_stride_,
            dilation=self.conv_dilation_,
        )

        # convolution transpose input layer
        self.convolution_transpose_layer_lists_ = [
            self.convolution_transpose_layer_type_(in_channels=self.conv_first_layer_in_channel_,
                                                   out_channels=self.conv_output_channels_[0],
                                                   kernel_size=self.conv_kernel_size_[0],
                                                   padding=self.conv_padding_[0],
                                                   stride=self.conv_stride_[0],
                                                   dilation=self.conv_dilation_[0],
                                                   groups=self.conv_groups_[0],
                                                   bias=self.conv_bias_[0],
                                                   padding_mode=self.conv_padding_modes_[0]
                                                   ),
            self.active_function_type_()
        ]

        for i in range(1, self.conv_transpose_n_times_):
            self.convolution_transpose_layer_lists_.append(
                self.convolution_transpose_layer_type_(in_channels=self.conv_output_channels_[i - 1],
                                                       out_channels=self.conv_output_channels_[i],
                                                       kernel_size=self.conv_kernel_size_[i],
                                                       padding=self.conv_padding_[i],
                                                       stride=self.conv_stride_[i],
                                                       dilation=self.conv_dilation_[i],
                                                       groups=self.conv_groups_[i],
                                                       bias=self.conv_bias_[i],
                                                       padding_mode=self.conv_padding_modes_[i]))
            # No active function after the last layer
            if i < self.conv_transpose_n_times_ - 1:
                self.convolution_transpose_layer_lists_.append(self.active_function_type_())
        self.conv_transpose_seq_ = nn.Sequential(*self.convolution_transpose_layer_lists_)

    def forward(self, x_sample: torch.Tensor):
        return self.conv_transpose_seq_(x_sample)

    predict = forward

    def summary(self) -> str:
        return "Convolution Transpose NDim NTime: " + str(self.properties_dict())

    def to_device(self, device: Union[str, torch.device]):
        self.device_ = device if isinstance(device, torch.device) else torch.device(device)
        self.conv_transpose_seq_.to(self.device_)

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

    def properties_dict(self) -> Dict[str, Any]:
        return {
            "conv_first_layer_in_channel_": self.conv_first_layer_in_channel_,
            "conv_transpose_n_times_": self.conv_transpose_n_times_,
            "conv_kernel_size_": self.conv_kernel_size_.copy(),
            "conv_output_channels_": self.conv_output_channels_.copy(),
            "conv_padding_": self.conv_padding_.copy(),
            "conv_stride_": self.conv_stride_.copy(),
            "conv_dilation_": self.conv_dilation_.copy(),
            "convolution_transpose_layer_type_": self.convolution_transpose_layer_type_,
            "conv_groups_": self.conv_groups_.copy(),
            "conv_bias_": self.conv_bias_.copy(),
            "conv_padding_modes_": self.conv_padding_modes_.copy(),
            "active_function_type_": self.active_function_type_,
            "convolution_transpose_layer_lists_": (layers := copy.deepcopy(self.convolution_transpose_layer_lists_)),
            "conv_transpose_seq_": nn.Sequential(*layers),
        }

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
            ret = ConvTransposeNDimNTimes.shape_after_conv_transpose(
                ret,
                kernel_size=kernel_size[i],
                padding=padding[i],
                stride=stride[i],
                dilation=dilation[i]
            )
        return ret

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.conv_transpose_seq_.parameters(recurse=recurse)
