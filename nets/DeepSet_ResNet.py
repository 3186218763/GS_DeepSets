import torch
import torch.nn as nn
from utools.Net_Tools import Integrated_Net
from nets.DeepSets import DeepSetModel


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_Model(nn.Module):
    def __init__(self):
        super(ResNet_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 输入通道数为1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self.make_layer(64, 64, stride=1)
        self.layer2 = self.make_layer(64, 128, stride=2)
        self.layer3 = self.make_layer(128, 256, stride=2)
        self.layer4 = self.make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 3)

    def make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # 在通道维度上添加额外的维度，使输入成为四维张量
        x = x.unsqueeze(1)  # 在第1个位置添加一个维度
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepSet_ResNet(nn.Module):
    def __init__(self, deepset_hidden_size: int, deepset_out_size: int, input_size: int = 6, Debug=False):
        super(DeepSet_ResNet, self).__init__()
        DeepSet = DeepSetModel(input_size=input_size,
                               output_size=deepset_out_size,
                               hidden_size=deepset_hidden_size,
                               Debug=Debug)
        ResNet = ResNet_Model()
        self.DeepSet_ResNet = Integrated_Net(DeepSet, ResNet)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
        out = self.DeepSet_ResNet(x)
        return out


# DeepSet_ResNet网络测试
if __name__ == '__main__':
    size = (64, 32, 6)
    tensor = torch.rand(size, dtype=torch.float32)
    model = DeepSet_ResNet(input_size=6, deepset_out_size=64, deepset_hidden_size=64)
    out = model(tensor)
    print(out.shape)
