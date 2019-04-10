# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

dependent_var_index = 2
# X = dataset.iloc[:, 1].values # the independent variable(s), but treated as an array not a matrix
X = dataset.iloc[:, 1:2].values # the independent variable(s), 
# ------------------------------- the upper bound, 2 here, is not included

y = dataset.iloc[:, dependent_var_index].values # the dependent variable


# encode categorical data --- DELETE IF UNNEEDED ----------------------------------------------
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
column_to_encode = ____Column__Number___
X[:, column_to_encode] = LabelEncoder().fit_transform(X[:, column_to_encode]) # encode the last colum (Countries)
# --- --- categorical_features : array of columns considered
X = OneHotEncoder(categorical_features = [column_to_encode]).fit_transform(X).toarray()
"""
# -----------------------------------------------------------------------------


# avoid the dummy variable trap --- DELETE IF UNNEEDED ----------------------------------------
"""
X = X[:, 1:] # remove the fist column
"""

# Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling, 
# --- not needed for simple and multiple linear regressions algorithms on R and Python
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""


# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg   = PolynomialFeatures(degree=8)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# visualize
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff')

# that didn't work,
# let's visualize with the poly reg
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff')
plt.show() # much better!



# (optional) if you want to have a high
# resolution grid, do this:
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff')
plt.show() # much better!


# Predicting a single result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a single result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
