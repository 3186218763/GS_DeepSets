


# 创建一个 Net_Snapshot 的实例


# 假设你有一个形状为 (batch_size, 32, 6) 的输入张量
batch_size = 16
input_tensor = torch.randn(batch_size, 32, 6)

# 创建一个形状为 (batch_size, 32) 的掩码，只保留行的掩码
pad_mask = torch.ones(batch_size, 32, dtype=torch.bool)
pad_mask[:, :3] = False  # 前三行
pad_mask[:, -3:] = False  # 后三行

# 通过 model 处理输入数据
output_tensor = model(input_tensor)

# 输出形状
print(output_tensor.shape)  # 应该输出 (batch_size, 3)
