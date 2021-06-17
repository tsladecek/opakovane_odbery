#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

from umap import UMAP

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from scripts.constants import ATTRIBUTES

from scripts.gridsearch import gridsearch 
from scripts.model_search_space import model_search_space

import shap

# %%
train = pd.read_csv("data/train.tsv", sep='\t')
val = pd.read_csv("data/validation.tsv", sep='\t')


train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

# SCALING
ss = StandardScaler()


train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)


# %%
reducer = UMAP(n_neighbors=5, min_dist=0.01, random_state=1618)

reducer.fit(train_X_s)

train_umap = reducer.transform(train_X_s)
val_umap = reducer.transform(val_X_s)


sns.scatterplot(x=val_umap[:, 0], y=val_umap[:, 1], hue=val_y)
sns.scatterplot(x=train_umap[:, 0], y=train_umap[:, 1], hue=train_y)

# %%
sns.scatterplot(x="Gestational age", y="first_ff", data=train_X)

# %% Regression
poly = PolynomialFeatures(degree=2)
train_poly = poly.fit_transform(train_X)

poly = PolynomialFeatures(degree=2)
val_poly = poly.fit_transform(val_X)

train_X_poly = ss.fit_transform(train_poly)
val_X_poly = ss.transform(val_poly)

# %%
reducer = UMAP(n_neighbors=5, min_dist=0.01, random_state=1618)

reducer.fit(train_X_poly)

train_umap = reducer.transform(train_X_poly)
val_umap = reducer.transform(val_X_poly)


sns.scatterplot(x=val_umap[:, 0], y=val_umap[:, 1], hue=val_y)
sns.scatterplot(x=train_umap[:, 0], y=train_umap[:, 1], hue=train_y)

# %%
