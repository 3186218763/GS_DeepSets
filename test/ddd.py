import torch.nn as nn
import torch

class PhiConv(nn.Module):
    def __init__(self, kernel_size: int, stride: int, input_size: int = 6, output_size: int = 512, num_layers: int = 5):
        super(PhiConv, self).__init__()

        # 创建卷积层和归一化层
        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(
                nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(output_size))
            input_size = output_size  # 更新输入通道数

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将通道维度移动到最后
        x = self.conv(x)
        return x.permute(0, 2, 1)  # 将通道维度移动回去

# 创建1x1卷积的PhiConv
phi_conv_1x1 = PhiConv(kernel_size=1, stride=1)

# 创建3x3卷积的PhiConv（步长为1）
phi_conv_3x3_1 = PhiConv(kernel_size=3, stride=1)

# 创建3x3卷积的PhiConv（步长为2）
phi_conv_3x3_2 = PhiConv(kernel_size=3, stride=2)

# 测试示例输入
input_data = torch.randn(64, 32, 6)
output_1x1 = phi_conv_1x1(input_data)
output_3x3_1 = phi_conv_3x3_1(input_data)
output_3x3_2 = phi_conv_3x3_2(input_data)

print("Output shape for 1x1 conv:", output_1x1.shape)
print("Output shape for 3x3 conv with stride 1:", output_3x3_1.shape)
print("Output shape for 3x3 conv with stride 2:", output_3x3_2.shape)





