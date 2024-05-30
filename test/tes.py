import torch
import torch.nn as nn



# 创建一个 AttentionWeightedCombination 的实例
input_size = 3  # 输入特征维度
output_size = 3  # 输出特征维度
attention_model = AttentionWeightedCombination(input_size, output_size)

# 假设你有一个形状为 (batch_size, 8, 3) 的输入张量
# 为了简化示例，这里使用随机张量代替
batch_size = 64
input_tensor = torch.randn(batch_size, 8, 3)

# 通过 attention_model 处理输入数据
output_tensor = attention_model(input_tensor)

# 输出形状
print(output_tensor.shape)  # 应该输出 (batch_size, 3)



