import numpy as np
import matplotlib.pyplot as plt

# 大写字母表示矩阵，小写字母表示向量
X = np.array([[6], [8], [10], [14], [18]])  # 表示列向量，维度（5,1）
# x=np.array([[6,8,10,14,18]]) 表示行向量，维度（1,5）
# x=np.array([[6,8,10,12,14],[1,2,3,4,5]]) 表示二维矩阵，维度为（2,5）
# y=np.array([1,2,3,4,5]) 表示数组，不是行向量也不是列向量，其维度为（5，）
# print(X.shape) （5,1）
X.reshape(-1, 1)
y=[7,9,13,17.5,18]

def plot(X,y):
    '''
    数据展示
    :param X: 披萨的直径
    :param y: 披萨的价格
    :return:
    '''
    plt.figure()
    plt.title('Pizza price plotted against diameter')
    plt.xlabel('Diameter in inches')
    plt.ylabel('Price on dollars')
    plt.plot(X,y,'k.') #这里使用了plot中的连用方法
    # 'k.'表示颜色为k，线的类型为'.'(点线）
    # [x],y,[fmt]，其中fmt的基本格式是'[color][marker][line]'
    plt.axis([0,25,0,25]) #四个参数，表示：x轴最小坐标，x轴最大坐标，y轴最小坐标，y轴最大坐标
    plt.grid(True) #生成网格线
    plt.show()

# plot(X,y)

from sklearn.linear_model import LinearRegression
model=LinearRegression() #创建一个估计器实例
model.fit(X,y)  #用数据拟合模型

test_pizza=np.array([[12]]) #需要预测的x
print(model.predict(test_pizza)[0])#输出价格,原来为[13]，用[0]表示输出这个值