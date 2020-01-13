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

sample = read_csv("abalone.csv", delimiter=",", names=["sex",
                                                       "length",
                                                       "diameter",
                                                       "height",
                                                       "whole_weight",
                                                       "shucked_weight",
                                                       "viscera_weight", 
                                                       "shell_weight", 
                                                       "rings"])
sample_validation = sample[3500:]
sample = sample[:3500]
#sample_validation = read_csv("abalone.csv", delimiter=",", names=["sex",
#                                                       "length",
#                                                       "diameter",
#                                                       "height",
#                                                       "whole_weight",
#                                                       "shucked_weight",
#                                                       "viscera_weight", 
#                                                       "shell_weight", 
#                                                       "rings"])
sample.describe()
sample_validation.describe()
N = len(sample)
N_valid = len(sample_validation)
print("{} samples to train and {} samples to validate".format(N, N_valid))
#%%
model_rf = RandomForestRegressor(oob_score=True, n_estimators=100000, criterion='mse')
model_rf.fit(sample.loc[:,"length":"shell_weight"],sample.rings)
print("Final oob score: ", model_rf.oob_score_)
prediccions = model_rf.predict(sample.loc[:,"length":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE Random Forest:", NMSE)
prediccions = model_rf.predict(sample_validation.loc[:,"length":"shell_weight"])
mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation Random Forest:", mean_square_error)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation Random Forest:", NMSE_val)