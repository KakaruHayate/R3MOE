import numpy as np

array1 = np.load('E:\\GeneralCurveEstimator-SSL\\data0311\\lengths.npy')
array2 = np.load('E:\\GeneralCurveEstimator-SSL\\datagrid\\lengths.npy')

concatenated_array = np.concatenate((array1, array2))

np.save('lengths.npy', concatenated_array)
