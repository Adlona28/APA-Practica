#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 00:32:43 2020

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
#%%
sizes = [2,5,16,32]
model_nnet = MLPRegressor(alpha=0,
                           activation="logistic", 
                           max_iter=100000,
                           solver='lbfgs')
trc = GridSearchCV(estimator=model_nnet, 
                   param_grid ={'hidden_layer_sizes':sizes},
                   cv=50,
                   return_train_score=True)
model_10CV = trc.fit(sample.loc[:,"length":"shell_weight"],sample.rings)
print(model_10CV.best_score_)
prediccions = model_10CV.predict(sample.loc[:,"length":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)

#%%
import time
model_nnet = MLPRegressor(hidden_layer_sizes=32,
                           alpha=0,
                           activation="logistic", 
                           learning_rate = 'constant',
                           solver='lbfgs')
model_nnet.learning_rate_init = 1e-3
model_nnet.max_iter = 5000
model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
print("Final loss 1st training module: ", model_nnet.loss_)
prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
MAE = np.sum(abs(sample_validation.rings - prediccions))/N
print("MAE on validation data before refining:", MAE)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation MLP before refining:", NMSE_val)
model_nnet.learning_rate_init = 1e-4
model_nnet.max_iter = 10000
time.sleep(10)
model_nnet.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
#print("Coeficients:", model_nnet.coefs_, "Biasis:", model_nnet.intercepts_)
print("Final loss: ", model_nnet.loss_)
prediccions = model_nnet.predict(sample.loc[:,"male":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)
prediccions = model_nnet.predict(sample_validation.loc[:,"male":"shell_weight"])
MAE = np.sum(abs(sample_validation.rings - prediccions))/N
print("MAE on validation data:", MAE)
MSE_valid = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation MLP:", MSE_valid)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation MLP:", NMSE_val)
#%%