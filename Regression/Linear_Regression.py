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
    f = (F - (sum(features) / n)) / max(features)
    t = (T - (sum(targets) / n)) / max(targets)
    features_ = np.append(features_, f)
    targets_ = np.append(targets_, t)
    # print(f"original:{F}, normalized:{res_F}")
    # print(f"original:{T}, normalized:{res_T}")
    # print("************************************")

features = features_
targets = targets_

# In[]

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
LR = 0.6
n = len(features)


# bias  = 0
# weight  = 0
# LR = 0
# n = len(features)


# In[]

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


# In[]


# OLS  {Equation Based Approach}
def NormalEquation_OLS(features, targets):
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
    print("bias (m0):", b)
    print("weight(m1):", m, '\n')
    plt.scatter(features, targets)
    plt.plot(features, m * features + b, label='Normal Eqn (OLS)', color='black')
    plt.legend()
    plt.show()


# In[]

# {Matrix Based Approach} (x'x)m = x'y
def NormalEquation_matrix(features, targets):
    x_mat = np.matrix(features)

    if x_mat.shape == (1, len(features)):
        x_mat = np.insert(x_mat, 0, np.ones(shape=len(features)), axis=0)
        y_mat = np.matrix(targets).transpose()
        x_mat_trans = x_mat.transpose()
        temp = x_mat
        x_mat = x_mat_trans
        x_mat_trans = temp

    else:
        x_mat = np.insert(x_mat, 0, np.ones(shape=len(features)), axis=1)
        y_mat = np.matrix(targets)
        temp = x_mat
        x_mat_trans = temp.transpose()

    x_mat_trans_CROSS_x_mat = x_mat_trans * x_mat
    x_mat_trans_CROSS_y_mat = x_mat_trans * y_mat

    m = np.linalg.inv(x_mat_trans_CROSS_x_mat) * x_mat_trans_CROSS_y_mat
    m0 = m[0].item()
    m1 = m[1].item()

    print("bias (m0):", m0)
    print("weight(m1):", m1, '\n')
    plt.scatter(features, targets)
    plt.plot(features, m0 + m1 * features, label='Normal Eqn (Matrix)', color='black')
    plt.legend()
    plt.show()


# In[]
GD = GradientDescent(features, targets, loss, bias, weight)
normalEqn_OLS = NormalEquation_OLS(features, targets)
normalEqn_Mat = NormalEquation_matrix(features, targets)


