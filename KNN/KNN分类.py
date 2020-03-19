import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])

y_train = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female']


def plot(X, y):
    '''
    数据可视化，作散点图
    :param X: X_train
    :param y: y_train
    :return:
    '''
    plt.figure()
    plt.title('Human Heights and Weights by Sex')
    plt.xlabel('Height in cm')
    plt.ylabel('Weight in kg')
    for i, x in enumerate(X_train):
        # 使用‘x’标记训练集中的男性，使用菱形标记女性
        plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
    plt.grid(1)
    plt.show()


# plot(X_train,y_train)

X = np.array([[155, 70]])
# 输出训练数据集中与实例的欧式距离
distances = np.sqrt(np.sum((X_train - X) ** 2, axis=1))
print(distances)

# argsort()函数把数组按着从小到大排序
# argsort()返回最小的三个数的下标的列表
nearest_neighbor_indices = distances.argsort()[:3]
print(nearest_neighbor_indices)  # [0 5 8]
# take函数，取出y_train对应下标的数
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)
print(nearest_neighbor_genders)

from collections import Counter

b = Counter(np.take(y_train, distances.argsort()[:3]))
# Counter是一个容器
# most_common函数可以输出最大的前n个值，以列表+元组的形式输出
# print(b.most_common(2))
# [('female', 2), ('male', 1)]
# print(b.most_common(1)[0])
# 输出('female',2)
print(b.most_common(1)[0][0])
# female
