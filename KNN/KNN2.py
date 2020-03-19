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

X_test=np.array([
    [168,65],
    [180,96],
    [160,52],
    [169,67]
])
y_test=['male','male','female','female']
y_test_binarized=lb.transform(y_test)
print('Binarized labels: %s' %y_test_binarized.T[0])
#y_test_binarized.T[0]为[1 1 0 0]这个list
#y_test_binarized的值为[[1],[1],[0],[0]]为(4,1)
#y_test_binarized.T为[[1 1 0 0]]转化为(1,4)
predictions_binarized=clf.predict(X_test)
print('Binarized predictions: %s' %predictions_binarized)
#结果为[0 1 0 0]
print('Predicted labels: %s' %lb.inverse_transform(predictions_binarized))

from sklearn.metrics import accuracy_score
print('Accuracy: %s' %accuracy_score(y_test_binarized,predictions_binarized))

from sklearn.metrics import precision_score #精准率：针对预测结果而言，预测为正的样本中，有多少为正（负预测为正）
from sklearn.metrics import recall_score #召回率：针对样本而言，有多少正确的样本被预测了（正预测为正负）
from sklearn.metrics import f1_score #F1得分
from sklearn.metrics import matthews_corrcoef #马修斯相关系数（MCC）
def Evaluation(test,prediction):
    '''
    模型的评估（精准率、召回率、F1得分：精准率和召回率的调和平均,MCC得分）
    :param test: 测试集
    :param prediction: 对测试集的预测结果
    :return:
    '''
    print('Precision: %s' % precision_score(test,prediction))
    print('Recall: %s' % recall_score(test,prediction))
    print('F1 score: %s' % f1_score(test,prediction))
    print('Matthews correlation coefficient: %s' %matthews_corrcoef(test,prediction))
Evaluation(y_test_binarized,predictions_binarized)

#使用sklearn的classification_report进行各种评估说明
from sklearn.metrics import classification_report
print(classification_report(y_test_binarized,predictions_binarized,target_names=['male'],labels=[1]))
#传参说明：
#y_test和y_pred全部为1维数组
#labels表示需要评估的标签名称，在数组中的1和0
#target_names为我们自己指定的名字，你可以叫male也可以叫别的

