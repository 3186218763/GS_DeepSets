import numpy as np
import matplotlib.pyplot as plt

# 加载数据
init_positions = np.load('../npy/init_positions.npy')
real_positions = np.load('../npy/real_positions.npy')
guess_positions = np.load('../npy/guess_positions.npy')

# 提取坐标（仅XY平面）
init_x, init_y = init_positions[:, 0], init_positions[:, 1]
real_x, real_y = real_positions[:, 0], real_positions[:, 1]
guess_x, guess_y = guess_positions[:, 0], guess_positions[:, 1]

# 更大的缩放比例因子（拉大距离）
scale_factor = 100.0

# 缩放坐标
init_x *= scale_factor
init_y *= scale_factor

real_x *= scale_factor
real_y *= scale_factor

guess_x *= scale_factor
guess_y *= scale_factor

# 创建图形
plt.figure(figsize=(12, 10))  # 增加图形尺寸

# 绘制散点图，设置透明度
plt.scatter(init_x, init_y, c='green', s=5, alpha=0.5, label='Initial Positions')  # alpha参数控制透明度
plt.scatter(real_x, real_y, c='red', s=5, alpha=0.5, label='Real Positions')
plt.scatter(guess_x, guess_y, c='blue', s=5, alpha=0.5, label='Guess Positions')

# 设置标签
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('ECEF Positions')

# 设置相同的比例
plt.gca().set_aspect('equal', adjustable='box')

# 显示图例
plt.legend()

# 保存图形到文件
plt.savefig('ecef_positions.png')

# 显示图形
plt.show()






