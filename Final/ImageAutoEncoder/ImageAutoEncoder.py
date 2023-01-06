from typing import Dict, Any, Union, Optional, Iterator

import numpy as np
import torch

from mklearn.convolution.ConvoNTime import ConvTransposeNDimNTimes, ConvNDimNTimes
from mklearn.core.learn_module import LearnModule
from mklearn.core.mk_types import *
from mklearn.core.mklearn_errors import ShapeError


class ImageAutoEncoder(LearnModule):

    def __init__(self,
                 image_size: NDArray,  # [Channel, Height, Width]
                 encoder_output_vector_dim: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__()
        if len(image_size) < 3 or len(image_size.shape) != 1:
            raise ShapeError(f"Image shape invalid: {image_size}")

        self.encoder_output_vector_dim_ = encoder_output_vector_dim \
            if encoder_output_vector_dim is not None else image_size[1] * image_size[2] // 2

        self.encoder_out_channels_ = v \
            if (v := kwargs.get("encoder_out_channels", None)) is not None else np.array([1, 2, 4, 8, 16])

        self.conv_kernel_size_ = v \
            if (v := kwargs.get("conv_kernel_size", None)) is not None else np.array([[3, 3] for _ in range(5)])
        self.conv_padding_ = v \
            if (v := kwargs.get("conv_padding", None)) is not None else np.array([[0, 0] for _ in range(5)])
        self.conv_transpose_kernel_size_ = v \
            if (v := kwargs.get("conv_transpose_kernel_size", None)) is not None else np.array(
            [[3, 3] for _ in range(5)])
        self.conv_transpose_padding_ = v \
            if (v := kwargs.get("conv_transpose_padding", None)) is not None else np.array([[0, 0] for _ in range(5)])
        self.encoder_conv_ = ConvNDimNTimes(
            input_dim=image_size,
            kernel_sizes=self.conv_kernel_size_,
            out_channels=self.encoder_out_channels_,
            paddings=self.conv_padding_,
            conv_n_times=5,
        )
        linear_layer_input_size = int(np.prod([
            self.encoder_out_channels_[-1],
            *self.encoder_conv_.shape_after_convolution_  # image_size
        ]))
        self.encoder_list_ = [
            self.encoder_conv_,
            torch.nn.Flatten(),
            torch.nn.Linear(linear_layer_input_size, self.encoder_output_vector_dim_)
        ]

        self.encoder_ = torch.nn.Sequential(
            *self.encoder_list_
        )

        self.decoder_conv_trans_ = ConvTransposeNDimNTimes(
            input_dim=np.array([
                self.encoder_conv_.conv_layer_output_channels_[-1],
                *self.encoder_conv_.shape_after_convolution_
            ]),
            conv_transpose_n_times=5,
            kernel_size=self.conv_kernel_size_,
            padding=self.conv_transpose_padding_
        )
        self.decoder_list_ = [
            torch.nn.Linear(self.encoder_output_vector_dim_, linear_layer_input_size),
            torch.nn.Unflatten(1, tuple([int(i) for i in [
                self.encoder_conv_.conv_layer_output_channels_[-1],
                *self.encoder_conv_.shape_after_convolution_
            ]])),
            self.decoder_conv_trans_
        ]
        self.decoder_ = torch.nn.Sequential(*self.decoder_list_)

    def to_device(self, device: Union[str, torch.device]):
        pass

    def summary(self) -> str:
        return repr(self.encoder_) + "\n" + repr(self.decoder_)

    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        return {
            "encoder_output_vector_dim": self.encoder_output_vector_dim_,
            "encoder_out_channels": self.encoder_out_channels_,
            "conv_kernel_size": self.conv_kernel_size_,
            "conv_padding": self.conv_padding_,
            "conv_transpose_kernel_size": self.conv_transpose_kernel_size_,
            "conv_transpose_padding": self.conv_transpose_padding_,
        }

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return list(self.encoder_.parameters(recurse)) + list(self.decoder_.parameters(recurse))


__all__ = [
    "ImageAutoEncoder"
]
if __name__ == '__main__':
    ae = ImageAutoEncoder(image_size=np.array([1, 28, 28]), encoder_output_vector_dim=128)
    data = torch.rand(1, 1, 28, 28)
    print(ae.encoder_conv_.conv_seq_(data).shape)
    # print(ae.encoder_)
    encoded = ae.encoder_(data)
    print(encoded.shape)
    decoded = ae.decoder_(encoded)
    print(decoded.shape)
    print(torch.nn.MSELoss()(data, decoded))
