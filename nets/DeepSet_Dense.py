import torch
import torch.nn as nn

from nets.DeepSets import DeepSetModel


class DenseModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=3):
        super(DenseModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        # 将输入张量展平为 (batch_size, 8 * 1024)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class DeepSet_Dense(nn.Module):
    def __init__(self, deepset_hidden_size: int, deepset_out_size: int, dense_hidden=None, Debug=False):
        super(DeepSet_Dense, self).__init__()
        if dense_hidden is None:
            dense_hidden = [512, 256, 128, 64]

        self.DeepSet_Dense = nn.Sequential(
            DeepSetModel(output_size=deepset_out_size, hidden_size=deepset_hidden_size, Debug=Debug),
            DenseModel(input_size=8 * deepset_out_size, hidden_sizes=dense_hidden),
        )

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        out = self.DeepSet_Dense(x)
        return out


# DeepSet_Dense网络测试
if __name__ == '__main__':
    size = (64, 32, 6)
    deepset_out_size = 512
    deepset_hidden_size = 512
    tensor = torch.rand(size, dtype=torch.float32)
    net = DeepSet_Dense(deepset_out_size=1024, deepset_hidden_size=512)
    out = net(tensor)
    print(out.shape)
