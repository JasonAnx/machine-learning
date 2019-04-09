# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encode categorical data -----------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X[:, 3] = LabelEncoder().fit_transform(X[:, 3]) # encode the first colum (Countries)
# problem here is that we are giving each country a secuential number, which would make them seem to have some order/priority
#so instead we are going to use dummy encoding:
''' [0] : array of columns considered'''
X = OneHotEncoder(categorical_features = [3]).fit_transform(X).toarray()
# -----------------------------------------------------------------------------


# avoid the dummy variable trap
X = X[:, 1:] # remove the fist column


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# build the optimal model using backward Elimination
import statsmodels.formula.api as sm 
X = np.append(arr= np.ones((50, 1)).astype(int), values= X, axis = 1 )

X_opt = X[:,[0, 1, 2, 3, 4, 5]] # optimal
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS

X_opt = X[:,[0, 1, 3, 4, 5]] # optimal
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS

X_opt = X[:,[0, 3, 4, 5]] # optimal
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS

X_opt = X[:,[0, 3, 5]] # optimal
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS

X_opt = X[:,[0, 3]] # optimal
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS

regressor_OLS.summary()












