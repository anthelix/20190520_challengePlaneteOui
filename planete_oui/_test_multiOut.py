from sklearn import linear_model
# multivariate input
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
# univariate output
Y = [0., 1., 2., 3.]
# multivariate output
Z = [[0., 1.], [1., 2.], [2., 3.], [3., 4.]]

# ordinary least squares
clf = linear_model.LinearRegression()
# univariate
clf.fit(X, Y)
clf.predict ([[1, 0.]])
# multivariate
clf.fit(X, Z)
clf.predict ([[1, 0.]])

# Ridge
clf = linear_model.BayesianRidge()
# univariate
clf.fit(X, Y)
clf.predict ([[1, 0.]])
# multivariate
clf.fit(X, Z)
clf.predict ([[1, 0.]])

# Lasso
clf = linear_model.Lasso()
# univariate
clf.fit(X, Y)
clf.predict ([[1, 0.]])
# multivariate
clf.fit(X, Z)
clf.predict ([[1, 0.]])
