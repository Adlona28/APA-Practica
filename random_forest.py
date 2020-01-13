#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:37:10 2020

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
from sklearn.ensemble import RandomForestRegressor
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
model_rf = RandomForestRegressor(oob_score=True, criterion='mse', n_estimators=5000)
#model_rf = GridSearchCV(estimator=model_rf, 
#                   param_grid ={'n_estimators':[1000]},
#                   return_train_score=True)
model_rf = model_rf.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
#model_rf = GridSearchCV(estimator=model_rf, 
#                   param_grid ={'n_estimators':[5000, 50000]},
#                   return_train_score=True)
model_rf = model_rf.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
#print("Best score: ", model_rf.best_score_)
#print("Number of trees:", model_rf.best_params_)

prediccions = model_rf.predict(sample.loc[:,"male":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE Random Forest:", NMSE)
prediccions = model_rf.predict(sample_validation.loc[:,"male":"shell_weight"])
MAE = np.sum(abs(sample_validation.rings - prediccions))/N
print("MAE on validation data:", MAE)
mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation Random Forest:", mean_square_error)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation Random Forest:", NMSE_val)