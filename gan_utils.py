import typing as tp


def get_padding(kernel_size, dilation=1):
    """
    计算一维卷积的填充大小。
    计算公式:
        padding = (kernel_size * dilation - dilation) / 2
    例如:
        kernel_size = 3, dilation = 1 => padding = 1
        kernel_size = 3, dilation = 2 => padding = 2

    参数说明:
        kernel_size (int): 卷积核大小。
        dilation (int, 可选): 膨胀因子，默认为1。

    返回:
        int: 计算得到的填充大小。
    """
    # 计算填充大小
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    """
    计算二维卷积的填充大小。
    计算公式:
        padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
        padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
    例如:
        kernel_size = (3, 3), dilation = (1, 1) => padding = (1, 1)
        kernel_size = (3, 3), dilation = (2, 2) => padding = (2, 2)

    参数说明:
        kernel_size (Tuple[int, int]): 卷积核大小，格式为 (高度, 宽度)。
        dilation (Tuple[int, int], 可选): 膨胀因子，默认为 (1, 1)。

    返回:
        Tuple[int, int]: 计算得到的二维填充大小，格式为 (高度填充, 宽度填充)。
    """
    # 返回填充大小
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2, # 计算高度方向的填充大小
        ((kernel_size[1] - 1) * dilation[1]) // 2, # 计算宽度方向的填充大小
    )


def init_weights(m, mean=0.0, std=0.01):
    """
    初始化模型权重。
    正态分布初始化:
        mean: 均值
        std: 标准差
    例如:
        mean = 0.0, std = 0.01 => 权重服从 N(0, 0.01) 分布

    参数说明:
        m (nn.Module): 要初始化的模型或层。
        mean (float, 可选): 初始化权重的均值，默认为0.0。
        std (float, 可选): 初始化权重的标准差，默认为0.01。

    说明:
        该函数遍历模型的所有层，如果层名称包含 "Conv"，则使用正态分布初始化其权重。
    """
    classname = m.__class__.__name__
    # 如果类名包含 "Conv"
    if classname.find("Conv") != -1:
        # 使用正态分布初始化权重
        m.weight.data.normal_(mean, std)
