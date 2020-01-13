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
from numpy.linalg import inv, svd, cond, pinv
from statsmodels.genmod.generalized_linear_model import GLM
pd.set_option('precision', 3)
from sklearn.linear_model import Ridge, RidgeCV


#Lectura de les dades de train i validation
sample = read_csv("train.csv", delimiter=",", names=["male",
                                                       "female",
                                                       "infant",
                                                       "length",
                                                       "diameter",
                                                       "height",
                                                       "whole_weight",
                                                       "shucked_weight",
                                                       "viscera_weight", 
                                                       "shell_weight", 
                                                       "rings"])
sample_validation = read_csv("test.csv", delimiter=",", names=["male",
                                                       "female",
                                                       "infant",
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
model = GLM.from_formula('rings ~ male + female + infant + length + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight', sample)
H= np.diag(model.exog@inv(model.exog.T@model.exog)@model.exog.T)

#Primer rang de lambdes on fer la CV
lambdas = 10**np.arange(-6,2,0.1)
ridge = RidgeCV(alphas=lambdas,normalize=True)
ridge.fit(sample.loc[:,'male':'shell_weight'],
          sample.rings)
print('\nLAMBDA=',ridge.alpha_)

#Segon rang de lambdes per trobar un valor més precis
lambdas = np.arange(0.0002,0.0005,0.000001)
ridge = RidgeCV(alphas=lambdas,normalize=True,store_cv_values=True)
ridge.fit(sample.loc[:,'male':'shell_weight'],
          sample.rings)
print('LAMBDA=',ridge.alpha_)
alpha = ridge.alpha_
rings_ridge_reg = Ridge(alpha=alpha,
                          normalize=True).fit(sample.loc[:,'male':'shell_weight'],
                                              sample.rings)
print("Parametres:\n", pd.DataFrame([rings_ridge_reg.intercept_,
              rings_ridge_reg.coef_[0],
              rings_ridge_reg.coef_[1],
              rings_ridge_reg.coef_[2],
              rings_ridge_reg.coef_[3],
              rings_ridge_reg.coef_[4],
              rings_ridge_reg.coef_[5],
              rings_ridge_reg.coef_[6],
              rings_ridge_reg.coef_[7],
              rings_ridge_reg.coef_[8],
              rings_ridge_reg.coef_[9],],
             index=['Intercept',"male", "female", "infant", "length","diameter","height", "whole_weight","shucked_weight","viscera_weight", "shell_weight"]))
prediccions = rings_ridge_reg.predict(sample_validation.loc[:,'male':'shell_weight'])

MAE = np.sum(abs(sample_validation.rings - prediccions))/N
print("MAE on validation data:", MAE)
mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("Validation MSE:", mean_square_error)

NMSE_valid = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation variance".format(R_squared))
resid = rings_ridge_reg.predict(sample.loc[:,'male':'shell_weight'])-sample.rings
LOOCV_ridge = np.sum( (resid/(1-H))**2) / N
print("LOOCV:", LOOCV_ridge)
R2_LOOCV_ridge = (1 - LOOCV_ridge*N/((N-1)*np.var(sample.rings)))*100
print("R2_LOOCV:", R2_LOOCV_ridge)
# %%