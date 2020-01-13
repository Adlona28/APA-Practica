#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Alex Iniesta i Adrià Lozano
"""

import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



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
ntrees = np.array(np.round(10**np.arange(1.2,3,0.1)),dtype=int)
oob_score = None
n_ideal = None
mse = None
n_ideal_mse = None
mses = []
oobs = []
for n_est in ntrees:
    print("Testing {} estimators...".format(n_est), end='\r')
    model_rf = RandomForestRegressor(oob_score=True, criterion='mse', n_estimators=n_est)
    model_rf = model_rf.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
    #Ens guardem el nombre amb millor oob
    if oob_score is None or oob_score < model_rf.oob_score_:
        oob_score = model_rf.oob_score_
        n_ideal = n_est
    #Per fer una doble verificació, ens guardem també el nombre amb millor mse
    prediccions = model_rf.predict(sample_validation.loc[:,"male":"shell_weight"])
    MSE = np.sum((sample_validation.rings - prediccions)**2)/N_valid
    if mse is None or mse > MSE:
        n_ideal_mse = n_est
        mse = MSE
    mses.append(MSE)
    oobs.append(model_rf.oob_score_)


n_trees = (n_ideal + n_ideal_mse)//2#Esperem que les dos n trobades siguin la mateixa, pero per si a alguna execució no ho son, 
#ens amb el valor mitjà 
print("Best config with {} and {} so {}".format(n_ideal, n_ideal_mse, n_trees))
#Plot per veure el progrés de oob i mse en funció dels estimators
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(ntrees, mses, 'r')
ax.plot(ntrees, list(map(lambda x: x*7, oobs)), 'g')

#Model que ofereix millors resultats
model_rf = RandomForestRegressor(oob_score=True, criterion='mse', n_estimators=n_trees)
model_rf = model_rf.fit(sample.loc[:,"male":"shell_weight"],sample.rings)
#Fem les prediccions sobre el dataset de train
prediccions = model_rf.predict(sample.loc[:,"male":"shell_weight"])
#Obtenim R-Squared sobre train
NMSE = sum((sample.rings - prediccions)**2)/((N-1)*np.var(sample.rings))
print("NMSE Random Forest:", NMSE)
R_squared = (1 - NMSE)*100
print("Our model explain the {}% of the train variance".format(R_squared))
#Fem les prediccions sobre el dataset de validacio
prediccions = model_rf.predict(sample_validation.loc[:,"male":"shell_weight"])
#Obtenim les mètriques que ens serviran per comparar els diferents models
MAE = np.sum(abs(sample_validation.rings - prediccions))/N_valid
print("MAE on validation data:", MAE)
mean_square_error = np.sum((sample_validation.rings - prediccions)**2)/N_valid
print("MSE validation Random Forest:", mean_square_error)
NMSE_val = sum((sample_validation.rings - prediccions)**2)/((N_valid-1)*np.var(sample_validation.rings))
print("NMSE validation Random Forest:", NMSE_val)
R_squared = (1 - NMSE_val)*100
print("Our model explain the {}% of the validation variance".format(R_squared))