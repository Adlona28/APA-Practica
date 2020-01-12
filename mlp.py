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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

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

sizes = [2*i for i in range(1,11)]
model_nnet = MLPClassifier(alpha=0,
                           activation="logistic", 
                           max_iter=1000)
trc = GridSearchCV(estimator=model_nnet, 
                   param_grid ={'hidden_layer_sizes':sizes,
                                'solver': ['lbfgs', 'adam', 'sgd']},
                   cv=50,
                   return_train_score=True)
model_10CV = trc.fit(sample.loc[:,"length":"shell_weight"],sample.rings)
print(model_10CV.best_score_)
prediccions = model_10CV.predict(sample.loc[:,"length":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)

#%%
model_nnet = MLPClassifier(hidden_layer_sizes=[50, 10, 8],
                           alpha=0,
                           activation="identity", 
                           max_iter=100000,
                           solver='lbfgs')
model_nnet.fit(sample.loc[:,"length":"shell_weight"],sample.rings)
print("Final loss: ", model_nnet.loss_)
print(model_nnet.coefs_,
      model_nnet.intercepts_)
prediccions = model_nnet.predict(sample.loc[:,"length":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE MLP:", NMSE)

prediccions = model_nnet.predict(sample.loc[:,"length":"shell_weight"])
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE validation MLP:", NMSE)
#%%