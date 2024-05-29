import torch
import torch.nn as nn

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
        batch_size, num_groups, channels, seq_len = x.shape  # (batch_size, num_groups, channels, seq_len)

        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_groups, seq_len, channels)  # (batch_size*num_groups, seq_len, channels)

        x = self.transformer_encoder(x)  # (batch_size*num_groups, seq_len, channels)

        x = x.mean(dim=1)  # (batch_size*num_groups, channels)
        x = x.view(batch_size, num_groups, -1)  # (batch_size, num_groups, channels)

        if self.deepsets_only:
            x = x.sum(dim=1)  # (batch_size, channels)
            x = self.fc(x)  # (batch_size, output_size)
        else:
            x = self.fc(x)  # (batch_size, num_groups, output_size)
            x = x.unsqueeze(-1).expand(-1, -1, -1, self.output_channels)  # (batch_size, num_groups, output_size, output_channels)
            x = x.permute(0, 1, 3, 2)  # (batch_size, num_groups, output_channels, output_size)

        return x

# 创建一个测试输入张量
batch_size = 64
num_groups = 32
output_size = 512
channels = 8
input_tensor = torch.randn(batch_size, num_groups, output_size, channels)



# 前向传播
output = model(input_tensor)
print("Output shape:", output.shape)


