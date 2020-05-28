#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:32:01 2020

@author: bmathew
"""


import pandas as pd

dataset = pd.read_csv('/Users/bmathew/Documents/anaconda/data and resources/005 - Regression/01Students.csv')
df = dataset.copy()
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# split data into training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)


# train model on training data
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, y_train)

# predict based on trained model
y_predict = std_reg.predict(x_test)

slr_score = std_reg.score(x_test, y_test)
slr_coefficient = std_reg.coef_
slr_intercept = std_reg.intercept_

#y = slr_intercept + (slr_coefficient * x)

####################################################

from sklearn.metrics import mean_squared_error
import math

slr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))


# plot the trendline
import matplotlib.pyplot as plt
plt.scatter(x_test, y_predict)
plt.ylim(ymin=0)
plt.show()