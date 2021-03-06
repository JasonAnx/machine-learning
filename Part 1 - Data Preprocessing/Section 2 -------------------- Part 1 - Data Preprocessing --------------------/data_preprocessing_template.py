# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

dependent_var_index = __your__index
X = dataset.iloc[:, :-1].values # the independent variable(s)
y = dataset.iloc[:, dependent_var_index].values # the dependent variable


# encode categorical data --- DELETE IF UNNEEDED ----------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
column_to_encode = ____Column__Number___
X[:, column_to_encode] = LabelEncoder().fit_transform(X[:, column_to_encode]) # encode the last colum (Countries)
# --- --- categorical_features : array of columns considered
X = OneHotEncoder(categorical_features = [column_to_encode]).fit_transform(X).toarray()
# -----------------------------------------------------------------------------


# avoid the dummy variable trap --- DELETE IF UNNEEDED ----------------------------------------
X = X[:, 1:] # remove the fist column


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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