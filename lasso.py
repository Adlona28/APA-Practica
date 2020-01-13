#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:14:29 2020

@author: alex
"""
import numpy as np
import pandas as pd
from pandas import read_csv
from numpy.random import normal
from numpy.linalg import inv, svd, cond, pinv
from scipy import stats
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.linear_model import LassoCV, Lasso


#Lectura de les dades en train i validation
sample = read_csv("abalone.csv", delimiter=",", names=["sex",
                                                       "length",
                                                       "diameter",
                                                       "height",
                                                       "whole_weight",
                                                       "shucked_weight",
                                                       "viscera_weight", 
                                                       "shell_weight", 
                                                       "rings"])
sample_validation = read_csv("abalone.csv", delimiter=",", names=["sex",
                                                       "length",
                                                       "diameter",
                                                       "height",
                                                       "whole_weight",
                                                       "shucked_weight",
                                                       "viscera_weight", 
                                                       "shell_weight", 
                                                       "rings"])

sample.describe()
sample_validation.describe()
N = len(sample)
N_valid = len(sample_validation)
print("{} samples to train and {} samples to validate".format(N, N_valid))

#Per comparar el model amb regressio lineal
model = GLM.from_formula('rings ~ length + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight', sample)
H= np.diag(model.exog@inv(model.exog.T@model.exog)@model.exog.T)
lasso =LassoCV(max_iter=100000)
lasso.fit(sample.loc[:,'length':'shell_weight'],sample.rings)
print('alpha=',lasso.alpha_)
alpha = lasso.alpha_
rings_lasso_reg =Lasso(alpha=alpha, max_iter=100000)
rings_lasso_reg.fit(sample.loc[:,'length':'shell_weight'],sample.rings)

prediccions = rings_lasso_reg.predict(sample_validation.loc[:,'length':'shell_weight'])

mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("Validation MSE:", mean_square_error)

NMSE_valid = sum((sample.rings - prediccions)**2)/((N_valid-1)*np.var(sample.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation data".format(R_squared))
print('Intercept:', rings_lasso_reg.intercept_)
print('length:', rings_lasso_reg.coef_[0])
print('diameter:', rings_lasso_reg.coef_[1])
print('height:', rings_lasso_reg.coef_[2])
print('whole_weight:', rings_lasso_reg.coef_[3])
print('shucked_weight:', rings_lasso_reg.coef_[4])
print('viscera_weight:', rings_lasso_reg.coef_[5])
print('shell_weight:', rings_lasso_reg.coef_[6])

resid = rings_lasso_reg.predict(sample.loc[:,'length':'shell_weight'])-sample.rings
LOOCV_lasso = np.sum( (resid/(1-H))**2) / N
print("LOOCV:", LOOCV_lasso)
R2_LOOCV_lasso = (1 - LOOCV_lasso*N/((N-1)*np.var(sample.rings)))*100
print("R2_LOOCV:", R2_LOOCV_lasso)


#%%