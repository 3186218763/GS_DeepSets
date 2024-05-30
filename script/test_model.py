import torch
import torch.nn as nn

class AttentionWeightedCombination(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionWeightedCombination, self).__init__()
        # 定义一个全连接层，用于计算注意力权重
        self.attention = nn.Linear(input_size, 1)
        # 定义一个全连接层，将加权后的输入进行映射，得到最终输出
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # 将输入张量加权求和，考虑输入顺序
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        # 使用全连接层进行映射
        result = self.fc(weighted_sum)
        return result

# 创建一个 AttentionWeightedCombination 的实例
input_size = 3  # 输入特征维度
hidden_size = 64  # 隐藏层特征维度
attention_model = AttentionWeightedCombination(input_size, hidden_size)

# 假设你有一个形状为 (batch_size, 8, 3) 的输入张量
# 为了简化示例，这里使用随机张量代替
batch_size = 64
input_tensor = torch.randn(batch_size, 8, 3)

# 通过 attention_model 处理输入数据
output_tensor = attention_model(input_tensor)

# 输出形状
print(output_tensor.shape)  # 应该输出 (batch_size, hidden_size)



