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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


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


#%% MLP Single layer
sizes = [2,5,16,32, 64, 128, 172, 256, 384, 512]
mse = None
size_ideal_mse = None
mses = []
for size in sizes:
    print("Testing {} estimators...".format(size), end='\r')
    model_nnet = MLPRegressor(hidden_layer_sizes=size,
                              alpha=0,
                               activation="logistic", 
                               max_iter=1000,
                               solver='lbfgs')
    model_nnet = model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
    #Ens guardem el nombre amb millor mse
    prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
    MSE = np.sum((sample_validation.rings - prediccions)**2)/N_valid
    if mse is None or mse > MSE:
        size_ideal_mse = size
        mse = MSE
    mses.append(MSE)


size = size_ideal_mse 
print("Best config with {}".format(size))

model_nnet = MLPRegressor(hidden_layer_sizes=size,
                          alpha=0,
                           activation="logistic", 
                           max_iter=1000,
                           solver='lbfgs')
model_nnet = model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
print(model_nnet.get_params(), file=open('coeficients/mlp_singlelayer', 'w'))

prediccions = model_nnet.predict(sample.loc[:,"male":"shell_weight"])
#Obtenim R-Squared sobre train
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)
R_squared = (1 - NMSE)*100
print("Our model explain the {}% of the train variance".format(R_squared))


#Obtenim les mètriques que ens serviran per comparar els diferents models
prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data:", MAE)
MSE_valid = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation MLP:", MSE_valid)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation MLP:", NMSE_val)
R_squared = (1 - NMSE_val)*100
print("Our model explain the {}% of the validation variance".format(R_squared))

#%% MLP MultiLayer
import time
model_nnet = MLPRegressor(hidden_layer_sizes=[128,  64, 32],
                           alpha=0,
                           activation="logistic", 
                           learning_rate = 'constant',
                           solver='lbfgs')
model_nnet.learning_rate_init = 1e-3
model_nnet.max_iter = 500
model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
print("Final loss 1st training module: ", model_nnet.loss_)
prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data before refining:", MAE)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation MLP before refining:", NMSE_val)
model_nnet.learning_rate_init = 1e-5
model_nnet.max_iter = 500
time.sleep(10)
model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
print(model_nnet.get_params(), file=open('coeficients/mlp_multilayer', 'w'))
#print("Coeficients:", model_nnet.coefs_, "Biasis:", model_nnet.intercepts_)
print("Final loss: ", model_nnet.loss_)
#Fem les prediccions sobre el dataset de train
prediccions = model_nnet.predict(sample.loc[:,"male":"shell_weight"])
#Obtenim R-Squared sobre train
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)
R_squared = (1 - NMSE)*100
print("Our model explain the {}% of the train variance".format(R_squared))
#Fem les prediccions sobre el dataset de validacio
prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
#Obtenim les mètriques que ens serviran per comparar els diferents models
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data:", MAE)
MSE_valid = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation MLP:", MSE_valid)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation MLP:", NMSE_val)
R_squared = (1 - NMSE_val)*100
print("Our model explain the {}% of the validation variance".format(R_squared))
#%%