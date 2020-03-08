import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr_model = LinearRegression(fit_intercept=False)
pr_model.fit(X_train_poly, y_train)
print(pr_model.coef_)
print('Linear regression (order 5) score is:', pr_model.score(X_test_poly, y_test))

xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = pr_model.predict(xx_poly)

plt.scatter(xx, yy_poly)
plt.scatter(X_test, y_test)
plt.title('Linear regression (order 5) result')
plt.show()

ridge_model = Ridge(alpha=1, normalize=False, fit_intercept=False)
ridge_model.fit(X_train_poly, y_train)
print(ridge_model.coef_)
print('Ridge regression (order 5) score is:', ridge_model.score(X_train_poly, y_train))

yy_ridge = ridge_model.predict(xx_poly)
plt.scatter(xx, yy_ridge)
plt.scatter(X_test, y_test)
plt.title('Ridge regression (order 5) result')
plt.show()
