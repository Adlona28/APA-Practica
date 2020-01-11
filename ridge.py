#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:44:06 2020

@author: alex
"""
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
pd.set_option('precision', 3)
from sklearn.linear_model import Ridge, RidgeCV

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
lambdas = 10**np.arange(-6,2,0.1)
ridge = RidgeCV(alphas=lambdas,normalize=True)
ridge.fit(sample.loc[:,'length':'shell_weight'],
          sample.rings)
print('\nLAMBDA=',ridge.alpha_)

lambdas = np.arange(0.0001,0.0005,0.000001)
ridge = RidgeCV(alphas=lambdas,normalize=True,store_cv_values=True)
ridge.fit(sample.loc[:,'length':'shell_weight'],
          sample.rings)
print('LAMBDA=',ridge.alpha_)
alpha = ridge.alpha_
rings_ridge_reg = Ridge(alpha=alpha,
                          normalize=True).fit(sample.loc[:,'length':'shell_weight'],
                                              sample.rings)

prediccions = rings_ridge_reg.predict(sample_validation.loc[:,'length':'shell_weight'])

mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("Validation MSE:", mean_square_error)

NMSE_valid = sum((sample.rings - prediccions)**2)/((N_valid-1)*np.var(sample.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation data".format(R_squared))
resid = rings_ridge_reg.predict(sample.loc[:,'length':'shell_weight'])-sample.rings
LOOCV_ridge = np.sum( (resid/(1-H))**2) / N
print("LOOCV:", LOOCV_ridge)
R2_LOOCV_ridge = (1 - LOOCV_ridge*N/((N-1)*np.var(sample.rings)))*100
print("R2_LOOCV:", R2_LOOCV_ridge)
# %%