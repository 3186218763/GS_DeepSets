import torch
import torch.nn as nn

from utools.Net_Tools import check_phi_permutation_invariance
from nets.DeepSets import DeepSetModel


class Net_Equivariant(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(Net_Equivariant, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden * 2),
            nn.ReLU(),
            nn.Linear(dim_hidden * 2, dim_hidden)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim_hidden * 2, 512),  # 输入维度需要是 dim_hidden * 2
            nn.ReLU(),
            nn.Linear(512, dim_output)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.fc1(x)  # x 的形状为 (batch_size, 8, hidden_dim)

        x_sum = x.sum(dim=1)  # 对第二维求和，形状变为 (batch_size, hidden_dim)
        x_avg = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)  # 对第二维求平均，形状变为 (batch_size, hidden_dim)

        # 将求和和求平均的结果拼接
        x = torch.cat([x_sum, x_avg], dim=-1)  # 拼接后形状为 (batch_size, hidden_dim * 2)

        # 输出层变换
        x = self.fc2(x)  # x 的形状应为 (batch_size, dim_output)

        return x


class DeepSet_Equivariant(nn.Module):
    def __init__(self, deepset_hidden_size: int, deepset_out_size: int, input_size: int = 15, Debug=False):
        super(DeepSet_Equivariant, self).__init__()

        self.net = nn.Sequential(
            DeepSetModel(input_size=input_size, hidden_size=deepset_hidden_size,
                         output_size=deepset_out_size, Debug=Debug),
            Net_Equivariant(dim_input=deepset_out_size, dim_output=3, dim_hidden=1024)
        )

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        out = self.net(x)
        return out


if __name__ == '__main__':
    size = (64, 32, 15)
    tensor = torch.rand(size, dtype=torch.float32)
    model = DeepSet_Equivariant(input_size=15, deepset_out_size=64, deepset_hidden_size=64)
    check_phi_permutation_invariance(model)
    out = model(tensor)
    print(out.shape)
