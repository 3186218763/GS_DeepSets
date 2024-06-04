import numpy as np



# 示例数据
llh = np.array([[3980581.210, -111.159, 4966824.456], [4121321.562, -32259.202, 4913866.486]])  # 预测的 ECEF 坐标 (x, y, z)
llh_gt = np.array([[3980582.210, -112.159, 4966825.456], [4121322.562, -32260.202, 4913867.486]])  # 实际的 ECEF 坐标 (x, y, z)

# 计算评分
score = calc_score(llh, llh_gt)
print(f"Score: {score}")




