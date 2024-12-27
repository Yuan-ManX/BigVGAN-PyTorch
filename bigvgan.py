import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from gan_utils import *
from norm2d import *
from gated_activation_unit import *
from snake import *
from act import *
from filter import *
from resample import *


# 定义 LeakyReLU
LRELU_SLOPE = 0.1


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock1 类实现了一个带有周期性非线性激活函数的残差块（Residual Block）。
    该模块通过堆叠多个卷积层和周期性非线性激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。
    该类支持使用 Snake 或 SnakeBeta 激活函数，并支持反锯齿处理。

    参数说明:
        cfg: 配置参数对象，包含以下字段:
            model:
                bigvgan:
                    snake_logscale (float): Snake 激活函数的 logscale 参数。
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int, int], 可选): 膨胀卷积的膨胀因子，默认为 (1, 3, 5)。
        activation (str, 可选): 激活函数类型，'snake' 或 'snakebeta'。默认为 None。
    """
    def __init__(
        self, cfg, channels, kernel_size=3, dilation=(1, 3, 5), activation=None
    ):
        super(AMPBlock1, self).__init__()
        # 存储配置参数
        self.cfg = cfg

        # 定义第一组卷积层列表
        self.convs1 = nn.ModuleList(
            [
                weight_norm(  # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs1.apply(init_weights)

        # 定义第二组卷积层列表
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs2.apply(init_weights)

        # 计算卷积层的总数
        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        if (
            activation == "snake"  # 如果激活函数类型为 "snake"
        ):  # 创建 Snake 激活函数列表，并应用反锯齿处理
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta" # 如果激活函数类型为 "snakebeta"
        ):  # 创建 SnakeBeta 激活函数列表，并应用反锯齿处理
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            # 如果激活函数类型不正确，则抛出错误
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量。
        """
        # 分别获取奇数层和偶数层的激活函数
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)  # 应用第一个激活函数
            xt = c1(xt)  # 通过第一个卷积层
            xt = a2(xt)  # 应用第二个激活函数
            xt = c2(xt)  # 通过第二个卷积层
            x = xt + x  # 残差连接

        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs1:
            # 移除第一组卷积层的权重归一化
            remove_weight_norm(l)
        for l in self.convs2:
            # 移除第二组卷积层的权重归一化
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock2 类实现了一个带有周期性非线性激活函数的残差块（Residual Block）。
    该模块通过堆叠多个卷积层和周期性非线性激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。
    该类支持使用 Snake 或 SnakeBeta 激活函数，并支持反锯齿处理。

    参数说明:
        cfg: 配置参数对象，包含以下字段:
            model:
                bigvgan:
                    snake_logscale (float): Snake 激活函数的 logscale 参数。
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int], 可选): 膨胀卷积的膨胀因子，默认为 (1, 3)。
        activation (str, 可选): 激活函数类型，'snake' 或 'snakebeta'。默认为 None。
    """
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        # 存储配置参数
        self.cfg = cfg

        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                 weight_norm(  # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs.apply(init_weights)

        # 计算卷积层的总数
        self.num_layers = len(self.convs)  # total number of conv layers

        if (
            activation == "snake" # 如果激活函数类型为 "snake"
        ):  # 创建 Snake 激活函数列表，并应用反锯齿处理
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"  # 如果激活函数类型为 "snakebeta"
        ):  # 创建 SnakeBeta 激活函数列表，并应用反锯齿处理
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            # 如果激活函数类型不正确，则抛出错误
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量。
        """
        # 遍历每个卷积层和激活函数
        for c, a in zip(self.convs, self.activations):
            # 应用激活函数
            xt = a(x)
            # 通过卷积层
            xt = c(xt)
            # 残差连接
            x = xt + x

        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs:
            # 移除卷积层的权重归一化
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    """
    BigVGAN模型类

    该模型是一个基于生成对抗网络（GAN）的语音生成模型，采用变分自编码器（VAE）架构。
    主要特点包括：
    - 使用多尺度卷积层进行上采样
    - 采用残差块（ResBlocks）增强模型深度和性能
    - 应用了权重归一化和激活函数（如Snake和SnakeBeta）
    """
    def __init__(self, cfg):
        """
        BigVGAN模型的初始化方法

        参数:
            cfg: 配置对象，包含模型的所有配置参数
        """
        super(BigVGAN, self).__init__()
        self.cfg = cfg

        # 获取残差块的内核大小列表的长度
        self.num_kernels = len(cfg.model.bigvgan.resblock_kernel_sizes)
        # 获取上采样率的列表的长度
        self.num_upsamples = len(cfg.model.bigvgan.upsample_rates)

        # 初始化预处理卷积层，用于提升通道数
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.preprocess.n_mel,                   # 输入通道数（梅尔频谱的维度）
                cfg.model.bigvgan.upsample_initial_channel,  # 输出通道数
                7,                                      # 卷积核大小
                1,                                      # 步幅
                padding=3                               # 填充
            )
        )

        # 根据配置选择残差块类型
        resblock = AMPBlock1 if cfg.model.bigvgan.resblock == "1" else AMPBlock2

        # 初始化上采样模块列表
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                cfg.model.bigvgan.upsample_rates,          # 上采样率列表
                cfg.model.bigvgan.upsample_kernel_sizes   # 上采样卷积核大小列表
            )
        ):
            # 为每个上采样阶段初始化一个卷积层
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                cfg.model.bigvgan.upsample_initial_channel // (2**i),
                                cfg.model.bigvgan.upsample_initial_channel
                                // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # 初始化残差块模块列表
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg.model.bigvgan.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    cfg.model.bigvgan.resblock_kernel_sizes,
                    cfg.model.bigvgan.resblock_dilation_sizes,
                )
            ):
                # 为每个残差块初始化一个残差块实例
                self.resblocks.append(
                    resblock(cfg, ch, k, d, activation=cfg.model.bigvgan.activation)
                )

        # 初始化后处理卷积层
        if cfg.model.bigvgan.activation == "snake":
            activation_post = Snake(ch, alpha_logscale=cfg.model.bigvgan.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif cfg.model.bigvgan.activation == "snakebeta":
            activation_post = SnakeBeta(
                ch, alpha_logscale=cfg.model.bigvgan.snake_logscale
            )
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        # 初始化最终的卷积层
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # 应用权重初始化
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        前向传播方法

        参数:
            x: 输入张量

        返回:
            输出张量
        """
        # 预处理卷积层
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                # 上采样
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    # 应用残差块
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对残差块输出进行平均
            x = xs / self.num_kernels

        # 后处理激活函数
        x = self.activation_post(x)
        # 最终卷积层
        x = self.conv_post(x)
        # Tanh激活
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """
        移除权重归一化

        该方法用于移除模型中所有权重归一化层的归一化，以稳定训练过程。
        """
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
