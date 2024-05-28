import torch
import torch.nn as nn
import torch.nn.functional as F


class PhiTransformer(nn.Module):
    def __init__(self, input_size: int = 6, output_size: int = 512, num_heads: int = 8, num_layers: int = 2,
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


# 创建Transformer的PhiTransformer
phi_transformer = PhiTransformer()

# 测试示例输入
input_data = torch.randn(64, 32, 6)
output_transformer = phi_transformer(input_data)

print("Output shape for Transformer:", output_transformer.shape)

