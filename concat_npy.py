import numpy as np

# 加载两个.npy文件
array1 = np.load('E:\\R3MOE-main\\data_0328\\lengths.npy')
array2 = np.load('E:\\R3MOE-main\\data_0714\\lengths.npy')

# 拼接数组
concatenated_array = np.concatenate((array1, array2))

# 可选：保存拼接后的数组为新的.npy文件
np.save('lengths.npy', concatenated_array)
