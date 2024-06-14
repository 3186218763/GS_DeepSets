import torch
import torch.nn as nn
import torch.nn.functional as F


class DSEN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DSEN, self).__init__()
        # 定义输入层、隐藏层和输出层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)  # 输出层输入为 hidden_dim * 2

        # 定义对称操作：全局求和和求平均
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x 的形状为 (batch_size, 8, 1024)

        # 输入层变换
        x = F.relu(self.input_layer(x))  # 输出形状 (batch_size, 8, hidden_dim)

        # 隐藏层变换
        x = F.relu(self.hidden_layer(x))  # 输出形状 (batch_size, 8, hidden_dim)

        # 对称操作
        x_sum = x.sum(dim=1)  # 对第二维求和，形状变为 (batch_size, hidden_dim)
        x_avg = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)  # 对第二维求平均，形状变为 (batch_size, hidden_dim)

        # 将求和和求平均的结果拼接
        x = torch.cat([x_sum, x_avg], dim=-1)  # 拼接后形状为 (batch_size, hidden_dim * 2)

        # 输出层变换
        x = self.output_layer(x)  # 输出形状 (batch_size, output_dim)

        return x


# 定义模型参数
input_dim = 1024
hidden_dim = 512
output_dim = 3

# 初始化模型
model = DSEN(input_dim, hidden_dim, output_dim)

# 创建一个示例输入
batch_size = 32
x = torch.randn(batch_size, 8, 1024)

# 前向传播
output = model(x)

# 输出结果
print(output.shape)  # 应输出 (batch_size, output_dim)







