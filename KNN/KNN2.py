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
# print(X_train.shape) (9,2)

y_train = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']


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


from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

lb = LabelBinarizer()
# 将对应的标签转化为0和1表示
y_train_binarized = lb.fit_transform(y_train)
# print(y_train_binarized)
# print(y_train_binarized.reshape(-1)) [1 1 1 1 0 0 0 0 0] 一个np.array

K = 3  # 设置3个分类
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
# reshape[行，列]
# reshape[-1,1]，-1在官网的解释为，未给定的，所以给出-1，表示行（列）不确定
# 给出1表示确定为1列，代码自动帮我们把数据给整理
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)
