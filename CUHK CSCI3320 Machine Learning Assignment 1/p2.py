import pandas as pd
import numpy as np

df = pd.read_csv('input_2.csv')
n = df.shape[0]
train = df.iloc[:int(n*0.8), :]
test = df.iloc[int(n*0.8):, :]
class_list = df['class'].unique()
class_list.sort()

prior = train['class'].value_counts(normalize=True)
prior_C1 = prior[1]
prior_C2 = prior[2]
print('Prior Probability of C1 \n\
P(C1) =', prior_C1)
print('Prior Probability of C2 \n\
P(C2) =', prior_C2)

m1 = train.loc[train['class'] == 1, 'feature_value'].mean()
s1 = train.loc[train['class'] == 1, 'feature_value'].std(ddof=0)
m2 = train.loc[train['class'] == 2, 'feature_value'].mean()
s2 = train.loc[train['class'] == 2, 'feature_value'].std(ddof=0)
print('Estimated mean for class 1 \n\
m1 =', m1)
print('Estimated variance for class 1 \n\
sigma square 1 =', s1**2)
print('Estimated mean for class 2 \n\
m2 =', m2)
print('Estimated variance for class 2 \n\
sigma square 2 =', s2**2)


def normal_pdf(mean, std, x):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mean)**2 / (2 * std**2))


def g1(x):
    return normal_pdf(m1, s1, x) * prior_C1


def g2(x):
    return normal_pdf(m2, s2, x) * prior_C2


test['g1(x)'] = test.feature_value.map(g1)
test['g2(x)'] = test.feature_value.map(g2)
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
