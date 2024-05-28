import torch
import torch.nn as nn
from utools.Net_Tools import Integrated_Net
from nets.DeepSets import DeepSetModel


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class DenseModel(nn.Module):
    def __init__(self, input_size=32, output_size=3):
        super(DenseModel, self).__init__()
        input_size = 32 * input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)
        return x


class DeepSet_Dense(nn.Module):
    def __init__(self,  deepset_hidden_size: int, deepset_out_size: int, output_size: int = 3, input_size: int = 6,
                 Debug=False):
        super(DeepSet_Dense, self).__init__()
        DeepSet = DeepSetModel(input_size=input_size, output_size=deepset_out_size, hidden_size=deepset_hidden_size, Debug=Debug)
        Dense = DenseModel(output_size=output_size, input_size=deepset_out_size)
        self.DeepSet_Dense = Integrated_Net(DeepSet, Dense)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        return self.DeepSet_Dense(x)


# DeepSet_Dense网络测试
if __name__ == '__main__':
    size = (64, 32, 6)
    tensor = torch.rand(size, dtype=torch.float32)
    net = DeepSet_Dense(input_size=6, output_size=3, deepset_out_size=64, deepset_hidden_size=64)
    out = net(tensor)
    print(out.shape)
    # DeepSet = DeepSetModel(input_size=6, output_size=64, hidden_size=32,
    #                        deepsets_only=False)
    # out = DeepSet(tensor)
    # print(out.shape)  # torch.Size([64, 32, 64])
    # Dense = DenseModel(output_size=3, input_size=64)
    # out2 = Dense(out)
    # print(out2.shape)
