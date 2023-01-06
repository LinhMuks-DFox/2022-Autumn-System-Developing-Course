from ._ConvNDimNTimes import ConvNDimNTimes
from ._ConvTransposeNDimNTimes import ConvTransposeNDimNTimes


def create_symmetries_conv_transpose(cov: ConvNDimNTimes,
                                     inverse_conv_property: bool = True) -> ConvTransposeNDimNTimes:
    if not isinstance(cov, ConvNDimNTimes):
        raise TypeError("Can not build ConvTransposeNDimNTimes instance from a non-ConvNDimNTimes instance.")

    return ConvTransposeNDimNTimes(
        input_dim=cov.conv_layer_output_channels_[-1],
        conv_transpose_n_times=cov.conv_n_times_,
        kernel_size=cov.conv_kernel_size_[::-1] if inverse_conv_property else cov.conv_kernel_size_,
        out_channels=cov.conv_layer_output_channels_[
                     ::-1] if inverse_conv_property else cov.conv_layer_output_channels_,
        padding=cov.conv_padding_[::-1] if inverse_conv_property else cov.conv_padding_,
        stride=cov.conv_stride_[::-1] if inverse_conv_property else cov.conv_stride_,
        dilation=cov.conv_dilation_[::-1] if inverse_conv_property else cov.conv_dilation_,
        convolution_transpose_layer_type=cov.convolution_layer_type_,
        groups=cov.conv_groups_[::-1] if inverse_conv_property else cov.conv_groups_,
        bias=cov.conv_bias_[::-1] if inverse_conv_property else cov.conv_bias_,
        padding_modes=cov.conv_padding_modes_[::-1] if inverse_conv_property else cov.conv_padding_modes_,
        active_function=cov.active_function_type_,
        device=cov.device_,
    )
