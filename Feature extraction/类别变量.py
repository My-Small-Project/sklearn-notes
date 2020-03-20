#从类别变量中提取特征
#类别变量是一组固定值，例如城市、性别等

from sklearn.feature_extraction import DictVectorizer
onehot_encoder=DictVectorizer()
X= [
{'city': 'New York'},
{'city': 'San Francisco'},
{'city': 'Chapel Hill'}
]
print(onehot_encoder.fit_transform(X))
    # (0, 1)	1.0
    # (1, 2)	1.0
    # (2, 0)	1.0
#用toarray()方法将结果转化为稀疏矩阵，这里的（0,1）对应着矩阵中的0,1位置，然后值为1
print(onehot_encoder.fit_transform(X).toarray())

#使用scale函数对特征进行标准化处理
from sklearn import preprocessing
import numpy as np
X = np.array([
[0., 0., 5., 13., 9., 1.],
[0., 0., 13., 15., 10., 15.],
[0., 3., 15., 2., 0., 11.]
])
print(preprocessing.scale(X))
