# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
\
dataset = pd.read_csv("Data.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


# taking care of missing data
from sklearn.preprocessing import Imputer
inputer = Imputer() # inspect the element to see the default parameters/values used
inputer.fit(x[:, 1:3])
x[:, 1:3] = inputer.transform(x[:, 1:3]) # replace the missing data by the mean of the column


# encode categorical data -----------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

x[:, 0] = LabelEncoder().fit_transform(x[:, 0]) # encode the first colum (Countries)
# problem here is that we are giving each country a secuential number, which would make them seem to have some order/priority
#so instead we are going to use dummy encoding:
''' [0] : array of columns considered'''
x = OneHotEncoder(categorical_features = [0]).fit_transform(x).toarray()

# be sure to use a new labelenconder for each transform, since they are fitted
y = LabelEncoder().fit_transform(y) # encode the first colum (Countries)
# -----------------------------------------------------------------------------

# splt the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)



























