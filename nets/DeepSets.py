import torch
import torch.nn as nn
from utools.Net_Tools import Integrated_Net
from utools.Mylog import logger


class PhiTransformer(nn.Module):
    def __init__(self, output_size: int, input_size: int = 6, num_heads: int = 8, num_layers: int = 2,
                 hidden_dim: int = 256):
        super(PhiTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, output_size)

        self.norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.input_proj(x)  # shape: (64, 32, hidden_dim)
        x = self.transformer_encoder(x)  # shape: (64, 32, hidden_dim)
        x = self.output_proj(x)  # shape: (64, 32, 512)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # shape: (64, 32, 512)

        return x


class PhiConv1x1(nn.Module):
    def __init__(self, output_size: int, input_size: int = 6, num_layers: int = 5):
        super(PhiConv1x1, self).__init__()

        # 创建卷积层和归一化层
        conv_layers = []
        for _ in range(num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(output_size)
            ])
            input_size = output_size  # 更新输入通道数

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将通道维度移动到最后
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # 将通道维度移动回来
        return x


class PhiFC(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32, Debug=False):
        super().__init__()

        self.Debug = Debug

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 8),
            nn.ReLU(),
            nn.Linear(hidden_size * 8, hidden_size * 16),
            nn.ReLU(),
            nn.Linear(hidden_size * 16, output_size)  # 将最后一层的输出大小改为 output_size
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PhiConv3x3(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3, input_size: int = 6, num_layers: int = 5):
        super(PhiConv3x3, self).__init__()

        # 创建卷积层和归一化层
        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(
                nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(output_size))
            input_size = output_size  # 更新输入通道数

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将通道维度移动到最后
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # 将通道维度移动回来
        return x


class SmallPhi(nn.Module):
    """
    这个是最终的Phi函数
    """

    def __init__(self, input_size, hidden_size=512):
        super(SmallPhi, self).__init__()

        self.phi_fc = PhiFC(input_size=input_size, output_size=hidden_size)
        self.phi_conv1x1 = PhiConv1x1(input_size=input_size, output_size=hidden_size)
        self.phi_conv3x3 = PhiConv3x3(input_size=input_size, output_size=hidden_size)
        self.phi_transformer = PhiTransformer(input_size=input_size, output_size=hidden_size)

    def forward(self, x):
        x_fc = self.phi_fc(x)
        x_conv1x1 = self.phi_conv1x1(x)
        x_conv3x3 = self.phi_conv3x3(x)
        x_transform = self.phi_transformer(x)
        out = torch.cat((x_fc.unsqueeze(3), x_conv1x1.unsqueeze(3), x_conv3x3.unsqueeze(3), x_transform.unsqueeze(3)),
                        dim=3)
        return out


class SmallRho(nn.Module):
    """
    This is the Rho function using self-attention for each channel separately, ensuring permutation invariance.
    """

    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, output_channels=4,
                 deepsets_only=False):
        super(SmallRho, self).__init__()
        self.deepsets_only = deepsets_only
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.output_channels = output_channels

    def forward(self, x):
        batch_size, num_groups, channels, seq_len = x.shape  # (64, 32, 512, 4)

        # Ensure the correct input shape for Transformer Encoder
        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_groups, seq_len, channels)  # (64*32, 4, 512)

        # Apply TransformerEncoder
        x = self.transformer_encoder(x)  # (64*32, 4, 512)

        # Ensuring permutation invariance by mean aggregation
        x = x.mean(dim=1)  # (64*32, 512)
        x = x.view(batch_size, num_groups, -1)  # (64, 32, 512)

        if self.deepsets_only:
            x = x.sum(dim=1)  # (64, 512)
            x = self.fc(x)  # (64, 3)
        else:
            x = self.fc(x)  # (64, 32, output_size)
            x = x.unsqueeze(-1).expand(-1, -1, -1, self.output_channels)  # (64, 32, output_size, output_channels)

        return x


class DeepSetModel(nn.Module):
    def __init__(self, input_size: int = 6, output_size: int = 256, hidden_size: int = 64, Debug=False):
        super().__init__()
        self.Debug = Debug

        phi = SmallPhi(input_size=input_size, hidden_size=hidden_size)
        rho = SmallRho(d_model=hidden_size, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                       output_size=output_size,output_channels=4,
                       deepsets_only=False)
        self.net = Integrated_Net(phi, rho)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
            if self.Debug:
                logger.debug(f"mask:{mask} x*mask:{x} ")
        out = self.net.forward(x)
        return out


class DeepSet_Only(nn.Module):
    def __init__(self, input_size: int = 6, output_size: int = 3, hidden_size: int = 64, Debug=False):
        super().__init__()
        self.Debug = Debug

        phi = SmallPhi(input_size=input_size, hidden_size=hidden_size)
        rho = SmallRho(d_model=hidden_size, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                       output_size=output_size,
                       deepsets_only=True)
        self.net = Integrated_Net(phi, rho)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x * pad_mask
            if self.Debug:
                logger.debug(f"mask:{mask} x*mask:{x} ")
        out = self.net.forward(x)
        return out


# DeepSets网络测试代码
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    size = (64, 32, 6)

    # 创建一个shape是size的tensor
    tensor = torch.rand(size, dtype=torch.float32)
    # 创建一个shape是size的掩码，所有元素都初始化为 1（即不需要掩码）
    mask = torch.ones(size, dtype=torch.float32)

    # 假设需要掩码的位置为前两行的前三列
    # 将这些位置的值设置为 0，表示这些位置需要被掩码
    mask[:, :2, :3] = 0

    # print(f"tensor的shape: {tensor.shape} tensor:{tensor}")

    # print(f"tensor的shape: {tensor.shape} tensor:{tensor}")

    # 如果需要Debug，将Debug设置为True。在logs/debug.log文件夹下查看日志
    # deepsets单独使用
    model1 = DeepSet_Only(input_size=6)
    out = model1(tensor, pad_mask=mask)
    print(f"DeepSet_Only的out的shape:{out.shape} ")

    # deepsets和其他net混合使用的情况
    model2 = DeepSetModel(input_size=6, output_size=256)
    out = model2(tensor, pad_mask=mask)
    print(f"DeepSetModel的out的shape:{out.shape} ")
