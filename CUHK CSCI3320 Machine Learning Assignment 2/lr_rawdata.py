import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn
from sklearn.preprocessing import StandardScaler
seaborn.set()


attributes = []
with open('imports-85.names', 'r') as f:
    n = f.read().find('Attribute:')
    f.seek(n)
    m = f.read().find('8. Missing Attribute Values:')
    f.seek(n)
    Attributes = f.read(m)
    for i in range(1, 27):
        start = Attributes.find(str(i) + '. ')
        end = Attributes.find(':', start)
        a = Attributes[start + 3: end].strip()
        attributes.append(a)

df = pd.read_csv('imports-85.data',
                 header=None,
                 names=[*attributes],
                 na_values='?')

df.dropna(inplace=True)
n = df.shape[0]
train = df.iloc[: int(n*0.8), :]
test = df.iloc[int(n*0.8):, :]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.array(train.horsepower).reshape(-1, 1))
X_test_scaled = scaler.transform(np.array(test.horsepower).reshape(-1, 1))
y_train_scaled = scaler.fit_transform(np.array(train.price).reshape(-1, 1))
y_test_scaled = scaler.transform(np.array(test.price).reshape(-1, 1))

LR = linear_model.LinearRegression()
LR.fit(X_train_scaled, y_train_scaled)
prediction = LR.predict(X_test_scaled)

plt.scatter(X_test_scaled, y_test_scaled, marker='o')
plt.scatter(X_test_scaled, prediction, marker='x')
plt.xlabel('Standardized horsepower')
plt.ylabel('Standardized price')
plt.title('Linear regression on cleaned and standardized test data')
plt.show()
