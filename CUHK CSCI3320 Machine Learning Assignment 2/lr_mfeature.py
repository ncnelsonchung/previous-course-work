import pandas as pd
import numpy as np
import seaborn
from sklearn import linear_model
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
        
df = pd.read_csv('imports-85.data', header=None, names=attributes, na_values=['?'])
df.dropna(inplace=True)
used_cols = ['city-mpg', 'horsepower', 'engine-size', 'peak-rpm', 'price']
df = df.loc[:, used_cols]
        
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.loc[:, ['city-mpg', 'horsepower', 'engine-size', 'peak-rpm']])
y_scaled = scaler.fit_transform(np.array(df.price).reshape(-1, 1))

X = np.hstack((np.ones([X_scaled.shape[0], 1]), X_scaled))
X = np.matrix(X)
y = np.matrix(y_scaled)
theta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print('Parameter theta calculated by normal equation:', tuple(np.array(theta_hat).ravel()))

SGDR = linear_model.SGDRegressor()
X = X_scaled
y = y_scaled.ravel()
SGDR.fit(X, y)
print('Parameter theta calculated by SGD:', tuple((SGDR.intercept_[0], *SGDR.coef_)))
