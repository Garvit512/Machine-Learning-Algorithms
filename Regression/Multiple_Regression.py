#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:43:39 2020

@author: garvit
"""

# Multivariate Linear Regression from scratch
'''
y = m0x0 + m1x1 + m2x2 + ....... 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('/home/garvit/Datasets/Multiple_Linear_Regression/50_Startups.csv')
corr = dataset.corr()
print(corr, '\n')

features = dataset.iloc[:, 0:4].values
targets = dataset.iloc[:, 4:5].values
n = len(features)

# Converting categories into numbers
labelEncoder = LabelEncoder()
features[:, 3] = labelEncoder.fit_transform(features[:, 3])

# One-hot encoding
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
                                remainder='passthrough'
                                )

features = transformer.fit_transform(features)

# Feature scaling
scaler = StandardScaler()
features = scaler.fit_transform(features)
targets = scaler.fit_transform(targets)

# Visulizing correlation
for index, data in enumerate(features.transpose()):
    plt.scatter(data, targets, label=f"x{index}")
    plt.legend(loc='upper left')
    plt.show()


# Multivariate Linear regression
def NormalEquation_matrix(features, targets):  # (matrix based) (x'x)m = x'y
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
        del (temp)

    x_mat_trans_CROSS_x_mat = x_mat_trans * x_mat
    x_mat_trans_CROSS_y_mat = x_mat_trans * y_mat

    m = np.linalg.inv(x_mat_trans_CROSS_x_mat) * x_mat_trans_CROSS_y_mat
    print("bias (m0):", m[0], '\n')
    print("weight (m1, m2...):", m[1:], '\n')


normalEqn_Mat = NormalEquation_matrix(features, targets)

