"""
Name / Data Type / Measurement Unit / Description
-----------------------------
Sex / nominal / -- / M, F, and I (infant)
length / continuous / mm / Longest shell measurement
diam / continuous / mm / diam
height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
rings / integer / -- / +1.5 gives the age in years


@authors: Alex Iniesta i Adrià Lozano
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
import os
import warnings
warnings.filterwarnings('ignore')

def grafiques(abalone):
    sns.set()

    abalone = pd.read_csv('abalone.csv', sep=',')
    abalone.head()
    abalone.describe()

    rows = 2
    cols = 2
    i = 0

    plt.figure(figsize=(cols * 5, rows * 5))

    i += 1
    plt.subplot(rows, cols, i)
    plt.xticks(range(0, 31, 4))
    plt.xlim(0, 30)
    _ = sns.distplot(abalone['rings'], kde=False, bins=range(0, 31, 2))

    i += 1
    plt.subplot(rows, cols, i)
    _ = sns.distplot(abalone['rings'])

    i += 1
    plt.subplot(rows, cols, i)
    plt.xticks(range(0, 31, 4))
    plt.xlim(0, 30)
    _ = sns.boxplot(abalone['rings'])
    plt.savefig('./graficas/rings.png')

    plt.figure(figsize=(15, 15))

    colors = sns.color_palette()

    lines = 3
    rows = 3
    i = 0

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['length'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['diam'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['height'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['length'], kde=False, bins=np.arange(0.0, 0.9, 0.05), color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['diam'], kde=False, bins=np.arange(0.0, 0.7, 0.05), color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['height'], kde=False, bins=10, color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['length'], color=sns.color_palette()[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['diam'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['height'], color=colors[i % 3])

    plt.savefig('./graficas/sizes.png')

    abalone = abalone[abalone['height'] < 0.4]

    plt.figure(figsize=(15, 15))

    colors = sns.color_palette()

    lines = 3
    rows = 3
    i = 0

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['length'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['diam'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['height'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['length'], kde=False, bins=np.arange(0.0, 0.9, 0.05), color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['diam'], kde=False, bins=np.arange(0.0, 0.7, 0.05), color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.distplot(abalone['height'], kde=False, bins=10, color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['length'], color=sns.color_palette()[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['diam'], color=colors[i % 3])

    i += 1
    plt.subplot(lines, rows, i)
    _ = sns.boxplot(abalone['height'], color=colors[i % 3])

    plt.savefig('./graficas/sizesWithoutOutliers.png')

    plt.figure(figsize=(10, 10))
    corr = abalone.corr()
    _ = sns.heatmap(corr, annot=True)

    plt.savefig('./graficas/correlation.png')

    i_abalone = abalone[abalone['rings'] < 10]
    plt.figure(figsize=(10, 10))
    corr = i_abalone.corr()
    _ = sns.heatmap(corr, annot=True)

    plt.savefig('./graficas/correlation<10.png')

    a_abalone = abalone[abalone['rings'] >= 10]
    plt.figure(figsize=(10, 10))
    corr = a_abalone.corr()
    _ = sns.heatmap(corr, annot=True)

    plt.savefig('./graficas/correlation>10.png')

def elimOutliers(abalone_df):
    z = np.abs(stats.zscore(abalone_df))
    indexes = np.where(z > 3)
    return dataset.drop(indexes[0])

def createTestAndTrain(partToTest, dataset):

    """
    msk = np.random.rand(len(dataset)) < partToTrain
    train = dataset[msk]
    test = dataset[~msk]
    """
    X = dataset.drop('rings', axis = 1)
    y = dataset['rings']
    standardScale = StandardScaler()
    standardScale.fit_transform(X)
    selectkBest = SelectKBest()
    X_new = selectkBest.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = partToTest)


    df1 = pd.DataFrame(X_train)
    x = df1.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1 = pd.DataFrame(x_scaled)

    df2 = pd.DataFrame(y_train)


    df3 = pd.DataFrame(X_test)
    x = df3.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df3 = pd.DataFrame(x_scaled)

    df4 = pd.DataFrame(y_test)


    df1.to_csv('df1.csv')
    df2.to_csv('df2.csv')
    df3.to_csv('df3.csv')
    df4.to_csv('df4.csv')

    with open("df1.csv", "r") as f:
        lines1 = f.readlines()
    with open("df2.csv", "r") as f:
        lines2 = f.readlines()
    with open("train.csv", "w") as f:
        for i in range(len(lines1)):
            if lines1[i].strip("\n") != ",0,1,2,3,4,5,6,7,8,9":
                borrar1 = True
                borrar2 = True
                fline1 = ''
                fline2 = ''

                for c in lines1[i]:
                    if borrar1:
                        if c == ',':
                            borrar1 = False
                    else:
                        if c != '\n':
                            fline1 = fline1 + c

                for c in lines2[i]:
                    if borrar2:
                        if c == ',':
                            fline2 = fline2 + c
                            borrar2 = False
                    else:
                        fline2 = fline2 + c
                f.write(fline1+fline2)
    with open("df3.csv", "r") as f:
        lines1 = f.readlines()
    with open("df4.csv", "r") as f:
        lines2 = f.readlines()
    with open("test.csv", "w") as f:
        for i in range(len(lines1)):
            if lines1[i].strip("\n") != ",0,1,2,3,4,5,6,7,8,9":
                borrar1 = True
                borrar2 = True
                fline1 = ''
                fline2 = ''

                for c in lines1[i]:
                    if borrar1:
                        if c == ',':
                            borrar1 = False
                    else:
                        if c != '\n':
                            fline1 = fline1 + c

                for c in lines2[i]:
                    if borrar2:
                        if c == ',':
                            fline2 = fline2 + c
                            borrar2 = False
                    else:
                        fline2 = fline2 + c
                f.write(fline1+fline2)
    os.remove("df1.csv")
    os.remove("df2.csv")
    os.remove("df3.csv")
    os.remove("df4.csv")

dataset = pd.read_csv("abalone.csv")
X = dataset.drop('rings', axis = 1)
grafiques(dataset)
dataset=elimOutliers(dataset)
createTestAndTrain(0.20, dataset)
