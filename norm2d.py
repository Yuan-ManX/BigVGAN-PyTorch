import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm
import einops


# 定义一组卷积归一化方法的集合
CONV_NORMALIZATIONS = frozenset(
    [
        "none",  # 无归一化
        "weight_norm",  # 权重归一化
        "spectral_norm",  # 谱归一化
        "time_layer_norm",  # 时间轴层归一化
        "layer_norm",  # 层归一化
        "time_group_norm",  # 时间轴组归一化
    ]
)


class ConvLayerNorm(nn.LayerNorm):
    """
    ConvLayerNorm 类实现了一个对卷积操作友好的层归一化（LayerNorm）模块。
    该模块在执行归一化之前，将通道维度移动到最后，然后在归一化之后将通道维度恢复到原始位置。
    这种方式使得层归一化在卷积神经网络中更加适用，特别是当通道维度不是最后一个维度时。

    参数说明:
        normalized_shape (int 或 List[int] 或 torch.Size): 归一化的形状。
        **kwargs: 其他传递给 nn.LayerNorm 的关键字参数。
    """

    def __init__(self, normalized_shape, **kwargs):
        """
        初始化 ConvLayerNorm 类实例。

        参数:
            normalized_shape (int 或 List[int] 或 torch.Size): 归一化的形状。
            **kwargs: 其他传递给 nn.LayerNorm 的关键字参数。
        """
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        """
        前向传播方法，执行层归一化操作。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 归一化后的输出张量。
        """
        # 将输入张量重塑为 (batch_size, 时间步, 其他维度...) 的格式
        x = einops.rearrange(x, "b ... t -> b t ...")
        # 执行层归一化
        x = super().forward(x)
        # 将输出张量重塑回原始的形状
        x = einops.rearrange(x, "b t ... -> b ... t")
        return


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    """
    对模块应用参数化归一化。

    参数:
        module (nn.Module): 要应用归一化的模块。
        norm (str, 可选): 归一化类型，默认为 "none"。

    返回:
        nn.Module: 应用归一化后的模块。
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        # 应用权重归一化
        return weight_norm(module)
    elif norm == "spectral_norm":
        # 应用谱归一化
        return spectral_norm(module)
    else:
        # 如果归一化类型不在上述列表中，则不需要重新参数化
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """
    获取适当的归一化模块。如果 causal 是 True，则确保返回的模块是因果归一化的，
    或者如果归一化不支持因果评估，则返回错误。

    参数:
        module (nn.Module): 输入模块。
        causal (bool, 可选): 是否使用因果归一化，默认为 False。
        norm (str, 可选): 归一化类型，默认为 "none"。
        **norm_kwargs: 归一化参数的关键字参数。

    返回:
        nn.Module: 适当的归一化模块。

    异常:
        AssertionError: 如果归一化类型不在 CONV_NORMALIZATIONS 中，则抛出断言错误。
        ValueError: 如果归一化类型为 "time_group_norm" 且 causal 为 True，则抛出值错误。
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        # 返回 ConvLayerNorm 模块
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            # 如果是因果归一化，则抛出错误
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        # 返回 GroupNorm 模块
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


class NormConv2d(nn.Module):
    """
    NormConv2d 类是一个包装类，围绕 Conv2d 和归一化模块提供一个统一的接口，
    以便在不同归一化方法之间实现统一的接口。

    参数说明:
        *args: 传递给 Conv2d 的位置参数。
        norm (str, 可选): 归一化类型，默认为 "none"。
        norm_kwargs (dict, 可选): 归一化参数的关键字参数，默认为空字典。
        **kwargs: 传递给 Conv2d 的其他关键字参数。
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__()
        # 对 Conv2d 模块应用参数化归一化
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        # 获取适当的归一化模块
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        # 记录归一化类型
        self.norm_type = norm

    def forward(self, x):
        """
        前向传播方法，执行卷积和归一化操作。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 归一化后的输出张量。
        """
        # 执行卷积操作
        x = self.conv(x)
        # 执行归一化操作
        x = self.norm(x)
        return x
