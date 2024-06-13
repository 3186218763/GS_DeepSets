import torch
import torch.nn as nn

from utools.Mylog import logger
from utools.Net_Tools import Integrated_Net

torch.backends.cudnn.enabled = False


class DotProduct(nn.Module):
    def __init__(self, input_size, output_size):
        super(DotProduct, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, num_groups, input_size = x.shape

        # Reshape x to (batch_size * num_groups, input_size)
        x = x.reshape(batch_size * num_groups, input_size)

        x = torch.matmul(x, self.weight) + self.bias  # (batch_size * num_groups, output_size)

        # Reshape x back (batch_size, num_groups, output_size)
        x = x.reshape(batch_size, num_groups, -1)

        x = torch.mean(x, dim=1)  # (batch_size, output_size)

        return x


class FC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, num_groups, input_size = x.shape

        # 逐元素操作
        x = self.fc(x)  # (batch_size, num_groups, hidden_size)
        x = torch.relu(x)

        # 对每个组执行均值池化，保持置换不变性
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size)

        # 最终输出层
        x = self.output_fc(x)  # (batch_size, output_size)

        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num_groups, input_size = x.shape
        x = x.reshape(batch_size * num_groups, input_size)  # Reshape to (batch_size * num_groups, input_size)
        x = self.model(x)  # Pass through the MLP layers
        x = x.reshape(batch_size, num_groups, -1)  # Reshape back to (batch_size, num_groups, output_size)
        x = torch.mean(x, dim=1)  # Aggregate over num_groups
        return x


class SAB(nn.Module):
    def __init__(self, input_size, nums_heads, output_size):
        super(SAB, self).__init__()
        self.output_size = output_size
        self.embed_dim = output_size * nums_heads
        self.mha = nn.MultiheadAttention(self.embed_dim, nums_heads)
        self.fc_q = nn.Linear(input_size, self.embed_dim)
        self.fc_k = nn.Linear(input_size, self.embed_dim)
        self.fc_v = nn.Linear(input_size, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 通过全连接层映射Q、K、V
        Q = self.fc_q(x.permute(1, 0, 2))  # 将输入形状变为 (seq_len, batch_size, embed_dim)，并进行映射
        K = self.fc_k(x.permute(1, 0, 2))
        V = self.fc_v(x.permute(1, 0, 2))

        # 使用自注意力
        out, _ = self.mha(Q, K, V)

        # 将输出形状变为 (batch_size, output_size)
        out = out.mean(dim=0)  # 对序列维度取平均
        out = self.fc_out(out)  # 应用输出全连接层

        return out


class Conv1x1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=1)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, input_size, seq_len)
        out = self.conv(x.permute(0, 2, 1))  # 将输入形状变为 (batch_size, seq_len, input_size)，以适应 Conv1d 的输入格式
        out = out.mean(dim=-1)  # 对 seq_len 维度取平均
        return out


class Pool(nn.Module):
    def __init__(self, output_size, input_size, pool_type: str, hidden_dim: int = 1024):
        super(Pool, self).__init__()

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

        x = x.reshape(batch_size * num_groups, channels)  # (batch_size * 32, input_size)
        x = self.input_proj(x)  # (batch_size * 32, hidden_dim)

        x = x.unsqueeze(-1)  # (batch_size * 32, hidden_dim, 1)
        x = self.pool(x)  # (batch_size * 32, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size * 32, hidden_dim)

        x = x.reshape(batch_size, num_groups, -1)  # shape: (batch_size, 32, hidden_dim)

        # 对 num_groups 维度进行池化
        x = x.mean(dim=1)  # shape: (batch_size, hidden_dim)

        x = self.output_proj(x)  # (batch_size, output_size)
        x = self.norm(x)  # (batch_size, output_size)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(Encoder, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (num_groups, batch_size, input_size)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, num_groups, input_size)
        attn_output = self.layer_norm(attn_output)
        attn_output = torch.mean(attn_output, dim=1)  # (batch_size, input_size)
        out = self.fc(attn_output)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        out = self.fc2(x)
        return out


class EncDec(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(EncDec, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_heads)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


class WeightedCombination(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeightedCombination, self).__init__()
        # 定义注意力层，用于计算注意力权重
        self.attention = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Softmax(dim=1)
        )
        # 定义一个全连接层，用于映射加权求和后的结果到输出大小
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # 计算注意力权重
        attention_weights = self.attention(x)
        # 加权求和
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        # 使用全连接层进行映射
        result = self.fc(weighted_sum)
        return result


class SmallPhi(nn.Module):
    """
    这个是最终的Phi函数
    """

    def __init__(self, input_size, output_size):
        super(SmallPhi, self).__init__()

        self.dot = DotProduct(input_size=input_size, output_size=output_size)
        self.fc = FC(input_size=input_size, output_size=output_size, hidden_size=2048)
        self.mlp = MLP(input_size=input_size, output_size=output_size, hidden_sizes=[512, 1024, 2048, 4096])
        self.sab = SAB(input_size=input_size, nums_heads=2, output_size=output_size)
        self.conv = Conv1x1(input_size=input_size, output_size=output_size)
        self.avgpool = Pool(input_size=input_size, output_size=output_size, pool_type="avg")
        self.maxpool = Pool(input_size=input_size, output_size=output_size, pool_type="max")
        self.encdec = EncDec(input_size=input_size, hidden_size=2048, output_size=output_size, num_heads=1)

    def forward(self, x):
        x_dot = self.dot(x)
        x_fc = self.fc(x)
        x_mlp = self.mlp(x)
        x_sab = self.sab(x)
        x_conv = self.conv(x)
        x_avgpool = self.avgpool(x)
        x_maxpool = self.maxpool(x)
        x_encdec = self.encdec(x)
        out = torch.cat((x_dot.unsqueeze(2), x_fc.unsqueeze(2), x_mlp.unsqueeze(2),
                         x_sab.unsqueeze(2), x_conv.unsqueeze(2), x_avgpool.unsqueeze(2),
                         x_maxpool.unsqueeze(2), x_encdec.unsqueeze(2)), dim=2)
        out = out.permute(0, 2, 1)

        return out


class SmallRho(nn.Module):
    def __init__(self, input_size, output_size, mode='direct'):
        super(SmallRho, self).__init__()
        self.mode = mode

        if mode == 'direct':
            self.feature_enhance = SmallPhi(input_size=input_size, output_size=3)
            self.weight_combination = WeightedCombination(input_size=3, output_size=3)
        elif mode == 'feature':
            self.feature_enhance = SmallPhi(input_size=input_size, output_size=output_size)

        else:
            raise Exception(f"没有这个模式：{mode}")

    def forward(self, x):
        if self.mode == 'direct':
            x = self.feature_enhance(x)
            out = self.weight_combination(x)

        elif self.mode == 'feature':
            out = self.feature_enhance(x)

        return out


class DeepSetModel(nn.Module):
    def __init__(self, input_size: int = 15, hidden_size: int = 1024, output_size: int = 1024, Debug=False):
        super().__init__()
        self.Debug = Debug

        self.net = nn.Sequential(
            SmallPhi(input_size=input_size, output_size=hidden_size),
            SmallRho(input_size=hidden_size, output_size=output_size*2, mode="feature"),
            SmallRho(input_size=output_size*2, output_size=output_size*2, mode="feature"),
            SmallRho(input_size=output_size*2, output_size=output_size, mode="feature"),
        )

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

        self.net = nn.Sequential(
            SmallPhi(input_size=input_size, output_size=2048),
            SmallRho(input_size=2048, output_size=1024, mode="feature"),
            SmallRho(input_size=1024, output_size=512, mode="feature"),
            SmallRho(input_size=512, output_size=256, mode="feature"),
            SmallRho(input_size=256, output_size=output_size, mode="direct")
        )

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
    size = (64, 32, 15)

    # 创建一个shape是size的tensor
    tensor = torch.rand(size, dtype=torch.float32)
    # 创建一个shape是size的掩码，所有元素都初始化为 1（即不需要掩码）
    mask = torch.ones(size, dtype=torch.float32)

    # 假设需要掩码的位置为前两行的前三列
    # 将这些位置的值设置为 0，表示这些位置需要被掩码
    mask[:, :2, :3] = 0

    # deepsets单独使用
    # model1 = DeepSet_Only(input_size=15, hidden_size=1024)
    # out = model1(tensor)
    # print(out.shape)

    # deepset混合使用
    # (64, 32, 6)
    net = DeepSetModel(input_size=15, output_size=512)
    out = net(tensor)
    print(out.shape)
