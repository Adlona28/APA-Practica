#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:53:39 2020

@author: alex
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


#Definició del model lineal com a combinació lineal de les entrades de les dades. 
model_glm = GLM.from_formula('rings ~ length + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight', sample)
model = model_glm.fit()
print(model.summary())

#Fem les prediccions de train
prediccions = model.predict(sample.loc[:, "length":"shell_weight"])

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
prediccions = model.predict(sample_validation.loc[:, "length":"shell_weight"])
#I obtenim les mètriques, com abans, pero sobre validation
MAE = np.sum(abs(sample_validation.rings - prediccions))/N
print("MAE on validation data:", MAE)

mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N
print("MSE on validation data:", mean_square_error)

NMSE_valid = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("Normalized MSE on Validation Data:", NMSE_valid)

#Calculem R_squared per fer-nos una idea de la accuracy del model
R_squared = (1 - NMSE_valid)*100
print("Our model explain the {}% of the validation data".format(R_squared))

H= np.diag(model_glm.exog@inv(model_glm.exog.T@model_glm.exog)@model_glm.exog.T)

LOOCV = np.sum( (model.resid_response/(1-H))**2) / N
print("LOOCV:", LOOCV)
R2_LOOCV = (1 - LOOCV*N/((N-1)*np.var(sample.rings)))*100

print("R2_LOOCV:", R2_LOOCV)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1,N_valid+1), prediccions, 'ro')
ax.plot(range(1,N_valid+1), 
       sample.rings, 'g^')

#"""
#Generarem models amb grau més gran que 1 per intentar millorar l'error obtingut 
#i anirem comprovant l'error obtingut en train i validation. Més concretament, 
#provarem els graus del 2 al 20 i estudiarem com escala l'error al augmentar aquest.
#"""

#coeffs = []
#nmse_train = [NMSE]
#nmse_valid = [NMSE_valid]
#models = [model]
#for i in range(3,22):
#    print("Generant model grau {}".format(i-1), end='\r')
#    model = GLM(sample.target, 
#                [np.vander(sample.input[j], i, increasing=True) for j in range(2)])
#    fmodel = model.fit()
#    coeffs.append(fmodel.params)
#    NMSE = fmodel.deviance/((N-1)*np.var(sample.target))
#    nmse_train.append(NMSE)
#    prediccions = fmodel.predict(exog=np.vander(sample_validation.input,
#                                              i+1, increasing=True))
#    NMSE_valid = sum((sample_validation.target - prediccions)**2)/((N_valid-1)*np.var(sample_validation.target))
#    print("For M = {}: NMSE Train={}, NMSE Validation={}".format(i-1, NMSE, NMSE_valid))
#    nmse_valid.append(NMSE_valid)
#    models.append(fmodel)
#
#fig = plt.figure(figsize=(100,50))
#ax = fig.add_subplot(5,5,1)
#ax.plot(range(1,21), nmse_train)
#plt.title("NMSE on training increasing degree")
#ax = fig.add_subplot(5,5,2)
#ax.plot(range(1,21), nmse_valid)
#plt.title("NMSE on validation increasing degree")
# %%