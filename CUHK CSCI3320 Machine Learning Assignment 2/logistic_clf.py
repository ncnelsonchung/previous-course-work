import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model
from sklearn.model_selection import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=19)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)
log_reg = linear_model.LogisticRegression()

log_reg.fit(X_train, y_train)
prediction = log_reg.predict(X_test)
print('Prediction contain value other than 0 or 1?', 'Yes' if set(np.unique(prediction)) in (0, 1) else 'No')

plt.scatter(X_test[prediction == 0, 0], X_test[prediction == 0, 1])
plt.scatter(X_test[prediction == 1, 0], X_test[prediction == 1, 1])
plt.title('Classification with Logistic Regression')
plt.show()

print('Number of wrong predictions is:', (prediction != y_test).sum())
