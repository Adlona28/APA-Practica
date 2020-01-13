#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Alex Iniesta i Adrià Lozano
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv, svd, cond, pinv
from pandas import read_csv
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('precision', 3)
from numpy.random import uniform, normal
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import pickle

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


#Definició del model lineal com a combinació lineal de les entrades de les dades. 
model_glm = GLM.from_formula('rings ~ male + female + infant + length + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight', sample)
model = model_glm.fit()
print(model.summary())
print(model.params, file=open('coeficients/linear_regression.txt', 'w'))
#Fem les prediccions de train
prediccions = model.predict(sample.loc[:, "male":"shell_weight"])

#Trobem les mètriques per evaluar el model, sobre les dades de train
MAE = np.sum(abs(sample.rings - prediccions))/N
print("MAE:", MAE)

mean_square_error = np.sum((sample.rings - prediccions)**2)/N
print("MSE:", mean_square_error)

root_mse = np.sqrt(model.deviance/N)
print("Root MSE:", root_mse)

NMSE = model.deviance/((N-1)*np.var(sample.rings))
print("Normalized MSE:", NMSE)

#Fem la prediccio sobre les dades de validation
prediccions = model.predict(sample_validation.loc[:, "male":"shell_weight"])
#I obtenim les mètriques, com abans, pero sobre validation
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data:", MAE)

mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE on validation data:", mean_square_error)

NMSE_valid = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)

#Calculem R_squared per fer-nos una idea de la capacitat del model
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation variance".format(R_squared))


#I mètriques addicionals que serveixen de comparació amb el model de regressió lineal
H= np.diag(model_glm.exog@inv(model_glm.exog.T@model_glm.exog)@model_glm.exog.T)

LOOCV = np.sum( (model.resid_response/(1-H))**2) / N
print("LOOCV:", LOOCV)
R2_LOOCV = (1 - LOOCV*N/((N-1)*np.var(sample_validation.rings)))*100

print("R2_LOOCV:", R2_LOOCV)
fig, ax = plt.subplots(figsize=(8,6))
nshow = 20
ax.plot(range(1,N_valid+1)[:nshow], prediccions[:nshow], 'ro')
ax.plot(range(1,N_valid+1)[:nshow], 
       sample_validation.rings[:nshow], 'g^')

# %%