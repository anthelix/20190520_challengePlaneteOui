#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:00:52 2019

@author: anthelix
"""


import pandas as pd
import numpy as np
from sklearn import linear_model

train = pd.DataFrame({"letter":["A", "B", "C", "D"], "value": [1, 2, 3, 4]})
y_train = train["value"]
X_train = train.drop(["value"], axis=1)
test = pd.DataFrame({"letter":["D", "D", "B", "E"], "value": [4, 5, 7, 19]})
X_test = test.drop(["value"], axis=1)
y_test = test["value"]
X_train.dtypes


all_data = pd.concat((X_train,X_test))
categorical_features = ['letter']
for column in categorical_features:
    X_train[column] = X_train[column].astype('category', categories = all_data[column].unique())
    X_test[column] = X_test[column].astype('category', categories = all_data[column].unique())


X_train = pd.get_dummies(X_train, drop_first=True)
print(X_train)


X_test = pd.get_dummies(X_test, drop_first=True)
print(X_train)
X_train.dtypes

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
model.score(X_test, y_test)
