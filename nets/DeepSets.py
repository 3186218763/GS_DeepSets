import torch
import torch.nn as nn
from utools.Net_Tools import Integrated_Net
from utools.Mylog import logger
import numpy as np

torch.backends.cudnn.enabled = False


class PhiDotProduct(nn.Module):
    def __init__(self, input_size, output_size):
        super(PhiDotProduct, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, x):
        return torch.matmul(x, self.weight)


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
    def __init__(self, input_size, output_size):
        super(PhiConv1x1, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 变换形状以适应卷积层 (batch_size, channels, seq_len)
        x = self.conv(x)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # 变回原来的形状


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


class PhiPool(nn.Module):
    def __init__(self, output_size: int, input_size: int = 6, pool_type: str = 'avg', hidden_dim: int = 256):
        super(PhiPool, self).__init__()

        self.input_proj = nn.Linear(input_size, hidden_dim)

        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("pool_type must be either 'avg' or 'max'")

        self.output_proj = nn.Linear(hidden_dim, output_size)
        self.norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        batch_size, num_groups, channels = x.shape  # (batch_size, 32, input_size)

        x = x.view(batch_size * num_groups, channels)  # (batch_size * 32, input_size)
        x = self.input_proj(x)  # (batch_size * 32, hidden_dim)

        x = x.unsqueeze(-1)  # (batch_size * 32, hidden_dim, 1)
        x = self.pool(x)  # (batch_size * 32, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size * 32, hidden_dim)

        x = self.output_proj(x)  # (batch_size * 32, output_size)
        x = self.norm(x)  # (batch_size * 32, output_size)

        x = x.view(batch_size, num_groups, -1)  # shape: (batch_size, 32, output_size)

        return x


class PhiPointwiseConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(PhiPointwiseConv, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_groups, input_size) -> (batch_size, input_size, num_groups)
        x = self.conv1x1(x)  # (batch_size, output_size, num_groups)
        x = self.norm(x)  # (batch_size, output_size, num_groups)
        x = x.permute(0, 2, 1)  # (batch_size, output_size, num_groups) -> (batch_size, num_groups, output_size)
        return x


class PhiAdditiveAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(PhiAdditiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(input_size, hidden_dim)
        self.K = nn.Linear(input_size, hidden_dim)
        self.V = nn.Linear(input_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size, num_groups, input_dim = x.shape  # (64, 32, 6)

        Q = self.Q(x)  # shape: (batch_size, num_groups, hidden_dim)
        K = self.K(x)  # shape: (batch_size, num_groups, hidden_dim)
        V = self.V(x)  # shape: (batch_size, num_groups, hidden_dim)

        # 计算注意力权重
        attention_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.hidden_dim),
                                          dim=2)  # shape: (batch_size, num_groups, num_groups)

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights, V)  # shape: (batch_size, num_groups, hidden_dim)

        output = self.output_proj(context_vector)  # shape: (batch_size, num_groups, output_size)

        return output


class SmallPhi(nn.Module):
    """
    这个是最终的Phi函数
    """

    def __init__(self, input_size, hidden_size=512):
        super(SmallPhi, self).__init__()

        self.phi_fc = PhiFC(input_size=input_size, output_size=hidden_size)
        self.phi_conv1x1 = PhiConv1x1(input_size=input_size, output_size=hidden_size)
        self.phi_maxPool = PhiPool(input_size=input_size, output_size=hidden_size, pool_type='max')
        self.phi_avgPool = PhiPool(input_size=input_size, output_size=hidden_size, pool_type='avg')
        self.phi_addAttention = PhiAdditiveAttention(input_size=input_size, output_size=hidden_size, hidden_dim=128)
        self.phi_dot = PhiDotProduct(input_size=input_size, output_size=hidden_size)
        self.phi_transformer = PhiTransformer(input_size=input_size, output_size=hidden_size)
        self.phi_pointWiseConv = PhiPointwiseConv(input_size=input_size, output_size=hidden_size)

    def forward(self, x):
        x_fc = self.phi_fc(x)
        x_conv1x1 = self.phi_conv1x1(x)
        x_maxPool = self.phi_maxPool(x)
        x_avgPool = self.phi_avgPool(x)
        x_addAttention = self.phi_addAttention(x)
        x_dot = self.phi_dot(x)
        x_transform = self.phi_transformer(x)
        x_pointWiseConv = self.phi_pointWiseConv(x)
        out = torch.cat((x_fc.unsqueeze(3), x_conv1x1.unsqueeze(3), x_maxPool.unsqueeze(3),
                         x_avgPool.unsqueeze(3), x_addAttention.unsqueeze(3), x_dot.unsqueeze(3),
                         x_transform.unsqueeze(3), x_pointWiseConv.unsqueeze(3)), dim=3)

        return out


class SmallRho(nn.Module):
    """
    把高维空间的特征映射出来
    """

    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, output_channels=9,
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

        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_groups, seq_len, channels)  # (64*32, 4, 512)

        x = self.transformer_encoder(x)  # (64*32, 4, 512)

        x = x.mean(dim=1)  # (64*32, 512)
        x = x.view(batch_size, num_groups, -1)  # (64, 32, 512)

        if self.deepsets_only:
            x = x.sum(dim=1)  # (64, 512)
            x = self.fc(x)  # (64, 3)
        else:
            x = self.fc(x)  # (64, 32, output_size)
            x = x.unsqueeze(-1).expand(-1, -1, -1, self.output_channels)  # (64, 32, output_size, output_channels)
            # 确保输出形状为 (batch, 32, output_size, channels)
            x = x.permute(0, 1, 3, 2)  # 将最后两个维度交换位置

        return x


class DeepSetModel(nn.Module):
    def __init__(self, input_size: int = 6, output_size: int = 256, hidden_size: int = 64, Debug=False):
        super().__init__()
        self.Debug = Debug

        phi = SmallPhi(input_size=input_size, hidden_size=hidden_size)
        rho = SmallRho(d_model=hidden_size, nhead=10, num_encoder_layers=24, dim_feedforward=2048,
                       output_size=output_size, output_channels=8,
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
