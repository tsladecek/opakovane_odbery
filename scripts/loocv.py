#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from scripts.constants import ATTRIBUTES

# %%
train = pd.read_csv(snakemake.input.train, sep='\t')
val = pd.read_csv(snakemake.input.validation, sep='\t')
test = pd.read_csv(snakemake.input.test, sep='\t')

# train = pd.read_csv('data/train.tsv', sep='\t')
# val = pd.read_csv('data/validation.tsv', sep='\t')
# test = pd.read_csv('data/test.tsv', sep='\t')

X_raw = pd.concat([train, val, test])
X_u = X_raw.loc[:, ATTRIBUTES]
y = X_raw.binary_2nd_result.values

# SCALING
ss = StandardScaler()
X = ss.fit_transform(X_u)


# %% LOOCV
yhat = []
for i in range(len(X)):
    X_train = np.concatenate((X[:i], X[(i + 1):]))
    y_train = np.concatenate((y[:i], y[(i + 1):]))
    
    X_test, y_test = X[i][np.newaxis, ...], y[i]
    
    m = XGBClassifier(random_state=1618, eval_metric='logloss')
    m.fit(X_train, y_train)
    
    yhat.append(m.predict(X_test))

# %%
(TN, FP), (FN, TP) = confusion_matrix(y, yhat)
acc = (TN + TP) / (TN + TP + FP + FN)
sens = TP / (TP + FN)
spec = TN / (TN + FP)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

metrics = pd.DataFrame({
    'accuracy': [acc],
    'sensitivity': [sens],
    'spec': [spec],
    'mcc': [mcc]
    })

# %%
metrics.to_csv(snakemake.output.loocv, sep='\t', index=False)
# metrics.to_csv('results/loocv.tsv', sep='\t', index=False)
