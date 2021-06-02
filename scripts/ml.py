#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from scripts.constants import ATTRIBUTES

from scripts.gridsearch import gridsearch 
from scripts.model_search_space import model_search_space

import shap

# %%
train = pd.read_csv("data/train.tsv", sep='\t', index_col=0)
val = pd.read_csv("data/validation.tsv", sep='\t', index_col=0)


train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

# %% SCALING
ss = StandardScaler()


train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)


# %% MODELS
results = []

for model in ["logisticregression", "lda", "qda", "svc", "randomforest", "xgboost", "knn"]:

    res, model = gridsearch(train_X, train_y, val_X, val_y, model,
                            model_search_space[model])
    
    results.append([model] + res.iloc[0, 1:].values.tolist())

