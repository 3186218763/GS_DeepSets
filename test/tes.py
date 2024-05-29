import torch
import torch.nn as nn


class MultiLayerFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerFC, self).__init__()
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # 隐藏层之间
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # 最后一层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平
        x = self.fc(x)
        return x


# 输入形状示例
batch_size = 64
input_channels = 4
input_rows = 32
output_size = 256
hidden_sizes = [512, 256, 128, 64]
final_output_size = 3

model = MultiLayerFC(input_rows * output_size * input_channels, hidden_sizes, final_output_size)
input_data = torch.randn(batch_size, input_rows, output_size, input_channels)





