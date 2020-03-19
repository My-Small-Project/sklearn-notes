# 根据一个人的性别和体重预测身高
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0]
])
y_train = [64, 86, 84, 80, 49, 59, 67, 54, 67]

X_test = np.array([
    [168, 1],
    [180, 1],
    [160, 0],
    [169, 0]
])
y_test = [65, 96, 52, 67]

K = 3
clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('Predicted weights: %s' % predictions)

# R2系数
print('Coefficient of determination: %s' % r2_score(y_test,
                                                    predictions))
# MAE是预测结果绝对误差的均值（y-yi）求和后求均值
print('Mean absolute error: %s' % mean_absolute_error(y_test,
                                                      predictions))
# MSE是均方误差（y-yi）的平方求和后求均值
print('Mean squared error: %s' % mean_squared_error(y_test,
                                                    predictions))

from scipy.spatial.distance import euclidean

# #heights in millimeters 特征缩放
# X_train = np.array([
# [1700, 1],
# [1600, 0]
# ])
# x_test = np.array([1640, 1]).reshape(1, -1)
# print(X_train,x_test)
# print(euclidean(X_train[0,:],x_test))
# print(euclidean(X_train[1,:],x_test))

# 对数据进行标准化处理
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# 将这些数据变成均值为0，方差为1的数据
# 均值归一化数据
X_train_scaled = ss.fit_transform(X_train)
print(X_train)
print(X_train_scaled)

X_test_scaled = ss.transform(X_test)

clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
print('Predicted wieghts: %s' % predictions)
print('Coefficient of determination: %s' % r2_score(y_test,
                                                    predictions))
print('Mean absolute error: %s' % mean_absolute_error(y_test,
                                                      predictions))
print('Mean squared error: %s' % mean_squared_error(y_test,
                                                    predictions))
