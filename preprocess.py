"""
Name / Data Type / Measurement Unit / Description
-----------------------------
Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter / continuous / mm / diam
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years
"""
import csv
import numpy as np
import time
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats

def grafiques(abalone_df):
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['length'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Longest shell measurement')
    fig.savefig('./graficas/lengthScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['diam'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Diameter (mm)')
    fig.savefig('./graficas/diamScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['height'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Height (mm)')
    fig.savefig('./graficas/heightScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['wholew'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Weight (grams)')
    fig.savefig('./graficas/wholewScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['shuckedw'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Shucked weight (grams)')
    fig.savefig('./graficas/shuckedwScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['visceraw'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Gut weight (mm)')
    fig.savefig('./graficas/viscerawScatter.png')
 
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(abalone_df['rings'], abalone_df['shellw'])
    ax.set_xlabel('Rings')
    ax.set_ylabel('Shell weight (mm)')
    fig.savefig('./graficas/shellwScatter.png')

def elimOutliers(abalone_df):
    z = np.abs(stats.zscore(abalone_df))
    indexes = np.where(z > 3)
    return dataset.drop(indexes[0])

def createTestAndTrain(partToTrain, dataset):
    msk = np.random.rand(len(dataset)) < partToTrain
    train = dataset[msk]
    test = dataset[~msk]
    train.to_csv('train.csv')
    test.to_csv('test.csv')

    with open("train.csv", "r") as f:
        lines = f.readlines()
    with open("train.csv", "w") as f:
        for line in lines:
            if line.strip("\n") != ",sex1,sex2,sex3,length,diam,height,wholew,shuckedw,visceraw,shellw,rings":
                borrar = True
                fline = ''
                for i in line:
                    if borrar:
                        if i == ',':
                            borrar = False
                    else:
                        fline = fline + i
                f.write(fline)

    with open("test.csv", "r") as f:
        lines = f.readlines()
    with open("test.csv", "w") as f:
        for line in lines:
            if line.strip("\n") != ",sex1,sex2,sex3,length,diam,height,wholew,shuckedw,visceraw,shellw,rings":
                borrar = True
                fline = ''
                for i in line:
                    if borrar:
                        if i == ',':
                            borrar = False
                    else:
                        fline = fline + i
                f.write(fline)

dataset = pd.read_csv("abalone.csv")
grafiques(dataset)
dataset=elimOutliers(dataset)
createTestAndTrain(0.8, dataset)