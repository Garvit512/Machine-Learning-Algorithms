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
import numpy as np # for random data generation
from sympy import symbols, Derivative # for partial derivative
import matplotlib.pyplot as plt # for visulization


#Generating Random Dataset
features = np.random.uniform(-4,4,500)       
targets = features + np.random.standard_normal(500)+2.5
n = len(features)

# File Data 
# dataset = pd.read_csv('/home/garvit/Datasets/years-of-experience-and-salary-dataset/Salary_Data.csv')

# features = dataset.iloc[:,0:1].values
# targets = dataset.iloc[:,1:].values
# n = len(features)


# # scaling data
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # sc = MinMaxScaler()
# sc = StandardScaler()
# features = sc.fit_transform(features)
# targets = sc.fit_transform(targets)


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


# loss function
loss = ((y-(c + m*x))**2) 


# derivative of loss function w.r.t y-intercept (c) and slope (m)
deriv_c = Derivative(loss, c).doit()
deriv_m = Derivative(loss, m).doit()


# setting initial/random weight and bias
bias  = 1
weight  = 0
LR = 0.09
n = len(features)


# Gradient Descent algorithm

def GradientDescent(features, targets, loss, bias, weight):
    for feature, target in zip(features, targets):
    
        deriv_c = Derivative(loss, c).doit()
        deriv_m = Derivative(loss, m).doit()
    
        loss_c = deriv_c.subs([(c, bias),(m, weight), (x, feature), (y, target)])/n    
        loss_m = deriv_m.subs([(c, bias),(m, weight), (x, feature), (y, target)])*feature/n
        
        updated_c = bias - LR*loss_c
        updated_m = weight - LR*loss_m
        bias = updated_c
        weight = updated_m
        
        # yy = float(bias + weight*features[sample])
    print("new bias:", bias)
    print("new weight:", weight,'\n')
    plt.scatter(features, targets)
    plt.plot(features, weight*features + bias, label = 'Gradient Descent', color = 'red')
    plt.legend()
    # plt.show()



def OLS(features, targets):
    x = features
    y = targets
    x_mean = sum(x)/len(x)
    y_mean = sum(y)/len(y)
    
    numerator_list = []
    denominator_list = []
    
    for feature, target in zip(x, y):
        x_minus_x_mean = feature - x_mean
        y_minus_y_mean = target - y_mean
        numerator = x_minus_x_mean*y_minus_y_mean
        denominator = x_minus_x_mean**2
        numerator_list.append(numerator)
        denominator_list.append(denominator)
    m = sum(numerator_list)/sum(denominator_list)
    b = y_mean - m*x_mean
    plt.scatter(features, targets)
    plt.plot(features, m*features + b, label = 'OLS',color = 'black')
    plt.legend()
    plt.show()

len(features)    
len(targets)




# GD = GradientDescent(features, targets, loss, bias, weight)

ols = OLS(features, targets)        
        
        
    
    














