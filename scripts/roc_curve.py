#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:06:59 2021

@author: tomas
"""

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn_json import from_json
from sklearn.svm import SVC
import shap

from scripts.constants import ATTRIBUTES

# %%
train = pd.read_csv("data/train.tsv", sep='\t', index_col=0)
val = pd.read_csv("data/validation.tsv", sep='\t', index_col=0)

train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

# SCALING
ss = StandardScaler()
train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)

val_dmat = xgb.DMatrix(val_X_s, val_y)

# %%
model = xgb.Booster()

model.load_model("results/models/xgboost.json")


# %%
svm = SVC(gamma=0.2, probability=True)

svm.fit(train_X, train_y)

val_yh = svm.predict(val_X)

print(np.mean(val_y == val_yh))

# %%
lda = from_json("results/models/randomforest.json")

# %%
e = shap.explainers.Exact(svm.predict_proba, train_X)
# shap.TreeExplainer(model)

sv = e(val_X)[:, :, 1]

# %%
shap.summary_plot(sv)

# %%
shap.force_plot(sv[0])

# %%
shap.plots.waterfall(sv[5])
