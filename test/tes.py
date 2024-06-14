

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
init_scores = np.random.rand(100)  # 生成100个随机分数作为示例
guess_scores = np.random.rand(100)



# 保存柱状图到文件
plt.savefig(f'score_mean.png')

# 显示图形
plt.show()



