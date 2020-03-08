import pandas as pd
import numpy as np

df = pd.read_csv('input_3.csv')
n = df.shape[0]
train = df.iloc[:int(n*0.8), :]
test = df.iloc[int(n*0.8):, :]
class_list = df['class'].unique()
class_list.sort()

prior_list = train['class'].value_counts(normalize=True)
for c in class_list:
    print('Prior Probability of class ' + str(c) + '\nP(C' + str(c) + ') =', prior_list[c])

mean_list = pd.Series()
std_list = pd.Series()
for c in class_list:
    mean_list.at[c] = train.loc[train['class'] == c, 'feature_value'].mean()
    std_list.at[c] = train.loc[train['class'] == c, 'feature_value'].std(ddof=0)
for c in class_list:
    print('Estimated mean for class ' + str(c) + '\nm' + str(c) + ' =', mean_list[c])
    print('Estimated variance for class ' + str(c) + '\nsigma square ' + str(c) + ' =', std_list[c]**2)


def normal_pdf(mean, std, x):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mean)**2 / (2 * std**2))


def discriminant(cl, x):
    return normal_pdf(mean_list[cl], std_list[cl], x) * prior_list[cl]


for c in class_list:
    test['g' + str(c) + '(x)'] = test.feature_value.map(lambda x: discriminant(c, x))
test['prediction'] = test.iloc[:, 2:].idxmax(axis=1).map(lambda x: int(x[1]))


def confusion_matrix(actual, prediction):
    confusion = pd.crosstab(actual, prediction)
    confusion.index.name = 'Actual'
    confusion.columns.name = 'Prediction'
    if set(confusion.columns.values) != set(class_list):
        miss_class = set(class_list).difference(set(confusion.columns.values))
        for cl in miss_class:
            confusion[cl] = 0
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
for c in class_list:
    print('Precision for class ' + str(c) + ' =', precision[c])
    print('Recall for class ' + str(c) + ' =', recall[c])
    print('f1 score for class ' + str(c) + ' =', f1[c])

print('Average f1 score =', f1_average)
