#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:32:46 2020

@author: garvit
"""

# Univariate Linear Regression from scratch
'''
y = mx + c 
'''

import math
import pandas as pd
import numpy as np  # for random data generation
from sympy import symbols, Derivative  # for partial derivative
import matplotlib.pyplot as plt  # for visulization

# Generating Random Dataset
# features = np.random.uniform(-4,4,500)
# targets = features + np.random.standard_normal(500)+2.5
# n = len(features)

# File Data
dataset = pd.read_csv('/home/garvit/Datasets/years-of-experience-and-salary-dataset/Salary_Data.csv')

features = dataset.iloc[:, 0:1].values
targets = dataset.iloc[:, 1:].values
n = len(features)

# Feature scaling (Mean Normalization)

# between -0.5 and 0.5
features_ = np.array([])
targets_ = np.array([])

for F, T in zip(features, targets):
    F_mean = sum(features) / n
    T_mean = sum(targets) / n

    F_std = np.std(features)
    T_std = np.std(targets)

    f = (F - F_mean) / F_std
    t = (T - T_mean) / T_std

    features_ = np.append(features_, f)
    targets_ = np.append(targets_, t)
    # print(f"original:{F}, normalized:{res_F}")
    # print(f"original:{T}, normalized:{res_T}")
    # print("************************************")

features = features_
targets = targets_

# Defining variables for partial derivative
m, x, c, y, LR, n = symbols('m, x, c, y, LR, n')

'''
m = slope
x = feature
y = target
c = y-intercept
LR = learning rate
n = no. of samples
'''

# loss function (MSE)
loss = ((y - (c + m * x)) ** 2)

# derivative of loss function w.r.t y-intercept (c) and slope (m)
deriv_c = Derivative(loss, c).doit()
deriv_m = Derivative(loss, m).doit()

# setting initial/random weight and bias
bias = 0
weight = 0
LR = 0.094
n = len(features)


# bias  = 0
# weight  = 0
# LR = 0
# n = len(features)


# Gradient Descent algorithm

def GradientDescent(features, targets, loss, bias, weight):
    loss_c_list = []
    loss_m_list = []
    for i in range(25):
        # plt.scatter(features, targets)
        for feature, target in zip(features, targets):
            # plt.scatter(features, targets)

            deriv_c = Derivative(loss, c).doit()
            deriv_m = Derivative(loss, m).doit()

            loss_c = deriv_c.subs([(c, bias), (m, weight), (x, feature), (y, target)])
            loss_m = deriv_m.subs([(c, bias), (m, weight), (x, feature), (y, target)]) * feature
            loss_c_list.append(loss_c)
            loss_m_list.append(loss_m)

            summation_loss_c = sum(loss_c_list) / (2 * n)
            summation_loss_m = sum(loss_m_list) / (2 * n)

        updated_c = bias - LR * summation_loss_c
        updated_m = weight - LR * summation_loss_m

        bias = updated_c
        weight = updated_m

        # yy = float(bias + weight*features[sample])
        # print("summation_loss_c", summation_loss_c)
        # print("summation_loss_m", summation_loss_m)
    print("new bias:", bias)
    print("new weight:", weight, '\n')
    print("-------------------------")
    # plt.scatter(features, targets)
    plt.plot(features, weight * features + bias, label='Gradient Descent', color='red')
    plt.legend()
    # plt.show()


def OLS(features, targets):
    x = features
    y = targets
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator_list = []
    denominator_list = []

    for feature, target in zip(x, y):
        x_minus_x_mean = feature - x_mean
        y_minus_y_mean = target - y_mean
        numerator = x_minus_x_mean * y_minus_y_mean
        denominator = x_minus_x_mean ** 2
        numerator_list.append(numerator)
        denominator_list.append(denominator)
    m = sum(numerator_list) / sum(denominator_list)
    b = y_mean - m * x_mean
    print("new bias:", b)
    print("new weight:", m, '\n')
    plt.scatter(features, targets)
    plt.plot(features, m * features + b, label='OLS', color='black')
    plt.legend()
    plt.show()


len(features)
len(targets)

GD = GradientDescent(features, targets, loss, bias, weight)

ols = OLS(features, targets)
