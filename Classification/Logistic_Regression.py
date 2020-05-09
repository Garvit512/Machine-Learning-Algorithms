"""
Created on Thu May 8 20:32:46 2020

@author: garvit
"""

# Univariate Logistic Regression from scratch
'''
y = 1/(1+e^-(m*x) )
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for visulization
from math import exp


# File Data
dataset = pd.read_csv('/Datasets/Logistic_Regression/Social_Network_Ads.csv')
features = dataset.iloc[:400, 3:4].values
targets = dataset.iloc[:400, 4:].values
n = len(features)


# Feature scaling (Mean Normalization)
# between -0.5 and 0.5
features_ = np.array([])

for F, T in zip(features, targets):
    f = (F - (sum(features) / n)) / max(features)
    features_ = np.append(features_, f)
features = features_


# loss function (Cross Entropy)
'''
loss = -y*log(y_pred) -(1-y)*log(1-y_pred)
'''

# setting initial/random weight and bias
bias = 0
weight = -30
LR = 0.03


# Gradient Descent algorithm
def GradientDescent(features, targets, weight):
    loss_m_list = []
    y_pred = []

    for i in range(30):
        for feature, target in zip(features, targets):

            x = feature
            y = target
            m = weight

            derivative_loss_wrt_m = (m*((1-y)*exp(m*x)+y))/(1+exp(m*x))
            loss_m_list.append(derivative_loss_wrt_m)
            summation_loss_m = sum(loss_m_list) / n

        updated_m = weight - LR * summation_loss_m
        weight = updated_m
        print("new weight:", weight, '\n')

    for i in (weight*features):
        zz = 1/(1+exp(-i))
        y_pred.append(zz)

    plt.scatter(features, targets)
    plt.plot(sorted(features), sorted(y_pred), label='Gradient Descent', color='red')
    plt.plot([0,0],[0,1])
    plt.show()

GD = GradientDescent(features, targets, weight)


