#用一个多重集对文本中的单词进行编码处理
corpus = [
'UNC played Duke in basketball',
'Duke lost the basketball game'
] #包含两个文档

#总共有8个不同的单词，第一个文档的第一个词是UNC，所以向量的第一个元素为1
#第一个文档没有包含game，所以特征向量的第8个元素为0
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
X = vectorizer.fit_transform(corpus) #建立字典表，需要先传入文档

print(vectorizer.get_feature_names()) #查看文档中的特征词
# ['basketball', 'duke', 'game', 'in', 'lost', 'played', 'the', 'unc']

print(vectorizer.vocabulary_) #查看词所对应的数字
# {'unc': 7, 'played': 5, 'duke': 1, 'in': 3, 'basketball': 0, 'lost': 4, 'the': 6, 'game': 2}

print(X.toarray())#转化为稀疏矩阵

corpus.append('I ate a sandwich')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
