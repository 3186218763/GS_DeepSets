import torch
import torch.nn as nn
import numpy as np


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


if __name__ == '__main__':
    phi_mlp = PhiAdditiveAttention(input_size=6, output_size=512, hidden_dim=128)
    size = (64, 32, 6)

    # 创建一个shape是size的tensor
    tensor = torch.rand(size, dtype=torch.float32)
    out = phi_mlp(tensor)
    print(out.shape)
