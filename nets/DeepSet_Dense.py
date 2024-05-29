import torch
import torch.nn as nn
from utools.Net_Tools import Integrated_Net
from nets.DeepSets import DeepSetModel


class DeepSetModel(nn.Module):
    def __init__(self, output_size, hidden_size, Debug=False):
        super(DeepSetModel, self).__init__()
        # 你的 DeepSetModel 实现
        self.fc = nn.Linear(6, hidden_size)  # 示例线性层
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 示例实现
        batch_size, num_groups, features = x.shape
        x = x.view(-1, features)  # (batch_size * num_groups, features)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        x = x.view(batch_size, num_groups, -1)  # 恢复形状 (batch_size, num_groups, output_size)
        x = x.unsqueeze(-1).repeat(1, 1, 1, 4)  # 加上通道维度 (batch_size, num_groups, output_size, 4)
        return x


class DenseModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        # 输入层到第一个隐藏层
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


class DeepSet_Dense(nn.Module):
    def __init__(self, deepset_hidden_size: int, deepset_out_size: int, output_size: int = 3,
                 dense_hidden=None, Debug=False):
        super(DeepSet_Dense, self).__init__()
        if dense_hidden is None:
            dense_hidden = [512, 256, 128, 64]
        self.DeepSet = DeepSetModel(output_size=deepset_out_size, hidden_size=deepset_hidden_size, Debug=Debug)
        # 修正 DenseModel 的输入尺寸
        input_size = 32 * deepset_out_size * 4
        self.Dense = DenseModel(input_size=input_size, hidden_sizes=dense_hidden, output_size=output_size)
        self.DeepSet_Dense = Integrated_Net(self.DeepSet, self.Dense)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        return self.DeepSet_Dense(x)


# DeepSet_Dense网络测试
if __name__ == '__main__':
    size = (64, 32, 6)
    tensor = torch.rand(size, dtype=torch.float32)
    net = DeepSet_Dense(deepset_hidden_size=64, deepset_out_size=1024, output_size=3)
    out = net(tensor)
    print(out.shape)  # 应该输出 (64, 3)
