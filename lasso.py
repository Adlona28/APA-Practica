#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Alex Iniesta i Adrià Lozano
"""
import numpy as np
from numpy.linalg import inv, svd, cond, pinv
from pandas import read_csv
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.linear_model import LassoCV, Lasso


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
#Apliquem CV per trobar la millor alpha
lasso =LassoCV(max_iter=100000)
lasso.fit(sample.loc[:,'male':'shell_weight'],sample.rings)
print('alpha=',lasso.alpha_)
alpha = lasso.alpha_
#Un cop trobada, definim el model de LASSO amb aquesta i fem el fit per definir el model
rings_lasso_reg =Lasso(alpha=alpha, max_iter=10000)
rings_lasso_reg.fit(sample.loc[:,'male':'shell_weight'],sample.rings)

print(rings_lasso_reg.coef_, file=open('coeficients/lasso', 'w'))
#Fem les prediccions sobre el dataset de validació
prediccions = rings_lasso_reg.predict(sample_validation.loc[:,'male':'shell_weight'])
#Obtenim les mètriques habituals
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data:", MAE)
mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("Validation MSE:", mean_square_error)

NMSE_valid = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation variance".format(R_squared))
#Mostrem els paràmetres del nostre model
print('Intercept:', rings_lasso_reg.intercept_)
print('male:', rings_lasso_reg.coef_[0])
print('female:', rings_lasso_reg.coef_[1])
print('infant:', rings_lasso_reg.coef_[2])
print('length:', rings_lasso_reg.coef_[3])
print('diameter:', rings_lasso_reg.coef_[4])
print('height:', rings_lasso_reg.coef_[5])
print('whole_weight:', rings_lasso_reg.coef_[6])
print('shucked_weight:', rings_lasso_reg.coef_[7])
print('viscera_weight:', rings_lasso_reg.coef_[8])
print('shell_weight:', rings_lasso_reg.coef_[9])

#I mètriques addicionals que serveixen de comparació amb el model de regressió lineal
resid = rings_lasso_reg.predict(sample.loc[:,'male':'shell_weight'])-sample.rings
LOOCV_lasso = np.sum( (resid/(1-H))**2) / N
print("LOOCV:", LOOCV_lasso)
R2_LOOCV_lasso = (1 - LOOCV_lasso*N/((N-1)*np.var(sample.rings)))*100
print("R2_LOOCV:", R2_LOOCV_lasso)


#%%