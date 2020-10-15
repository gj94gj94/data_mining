import pandas as pd
# Load origin dataset by pandas from UCI, load dataset-'abalone'
origin = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', names=['sex','length','diameter','height','whole','shucked','viscera','shell','rings'])
from sklearn.preprocessing import OneHotEncoder as ohe
import numpy as np
enc = ohe(categories='auto')
sex = origin['sex']
print(sex.to_numpy().reshape(-1,1))
t = enc.fit_transform(sex.to_numpy().reshape(-1,1)).toarray()
print(t)
sex = pd.DataFrame(data=t)
print(sex)
#origin = origin.drop('sex', axis=1).combine(sex, np.minimum, overwrite=False)
origin = pd.concat([origin.drop('sex', axis=1), sex], axis=1)
print(origin)


# Split origin dataset to training and testing dataset
from sklearn.model_selection import train_test_split as tts
train, test = tts(origin, test_size=0.2)
# Split origin dataset to value and label
trainX = train.drop('rings', axis=1).sort_index(ascending=True)
trainY = train['rings'].sort_index(ascending=True)
testX = test.drop('rings', axis=1).sort_index(ascending=True)
testY = test['rings'].sort_index(ascending=True)
print(testX, testY)

from sklearn import tree
dt = tree.DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=2)
dt = dt.fit(trainX, trainY)
train_predict = dt.predict(trainX)
test_predict = dt.predict(testX)

with open('abalone.dot', 'w') as f:
    f = tree.export_graphviz(dt, out_file=f)

from sklearn import metrics
train_acc = metrics.accuracy_score(trainY, train_predict)
test_acc = metrics.accuracy_score(testY, test_predict)
print('Accuracy on training data is %f' % train_acc)
print('Accuracy on testing data is %f' % test_acc)
