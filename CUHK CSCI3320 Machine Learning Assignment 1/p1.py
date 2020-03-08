import pandas as pd
import numpy as np

df = pd.read_csv('input_1.csv')
n = df.shape[0]
train = df.iloc[:int(n*0.8), :]
test = df.iloc[int(n*0.8):, :]
class_list = df['class'].unique()
class_list.sort()

table = pd.crosstab(train.feature_value, train['class'], margins=True)
prior_C1 = table.loc['All', 1] / table.loc['All', 'All']
prior_C2 = table.loc['All', 2] / table.loc['All', 'All']
p1 = table.loc[1, 1] / table.loc['All', 1]
p2 = table.loc[1, 2] / table.loc['All', 2]
print('Prior Probability of class 1 \n\
P(C1) =', prior_C1)
print('Prior Probability of class 2 \n\
P(C2) =', prior_C2)
print('Estimated probability of having an outcome 1 for class 1 \n\
p1 =', p1)
print('Estimated probability of having an outcome 1 for class 2 \n\
p2 =', p2)


def g1(x):
    return p1**x * (1-p1)**(1-x) * prior_C1


def g2(x):
    return p2**x * (1-p2)**(1-x) * prior_C2


test['g1(x)'] = g1(test.feature_value)
test['g2(x)'] = g2(test.feature_value)
test['prediction'] = np.where(test['g1(x)'] > test['g2(x)'], 1, 2)


def confusion_matrix(actual, prediction):
    confusion = pd.crosstab(actual, prediction)
    confusion.index.name = 'Actual'
    confusion.columns.name = 'Prediction'
    if set(confusion.columns.values) != set(class_list):
        miss_class = set(class_list).difference(set(confusion.columns.values))
        for c in miss_class:
            confusion[c] = 0
    else:
        pass
    confusion.sort_index(axis=1, inplace=True)
    return confusion


def accuracy_score(actual, prediction):
    confusion = confusion_matrix(actual, prediction)
    confusion = np.matrix(confusion)
    return float(confusion.trace() / confusion.sum())


def precision_score(actual, prediction):
    confusion = confusion_matrix(actual, prediction)
    temp = np.diag(confusion) / confusion.sum(axis=0)
    return pd.Series(temp, index=confusion.index.values)


def recall_score(actual, prediction):
    confusion = confusion_matrix(actual, prediction)
    temp = np.diag(confusion) / confusion.sum(axis=1)
    return pd.Series(temp, index=confusion.index.values)


def f1_score(actual, prediction, average=False):
    p = precision_score(actual, prediction)
    r = recall_score(actual, prediction)
    f = 2 * (p * r) / (p + r)
    if average:
        return f.mean()
    else:
        return f


confusion_m = confusion_matrix(test['class'], test.prediction)
print(confusion_m)

accuracy = accuracy_score(test['class'], test.prediction)
precision = precision_score(test['class'], test.prediction)
recall = recall_score(test['class'], test.prediction)
f1 = f1_score(test['class'], test.prediction, average=False)
f1_average = f1_score(test['class'], test.prediction, average=True)
print('Accuracy for all classes =', accuracy)
print('Precision for class 1 =', precision[1])
print('Precision for class 2 =', precision[2])
print('Recall for class 1 =', recall[1])
print('Recall for class 2 =', recall[2])
print('f1 score for class 1 =', f1[1])
print('f1 score for class 2 =', f1[2])
print('Average f1 score=', f1_average)
