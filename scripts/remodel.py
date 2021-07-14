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
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
from scripts.constants import ATTRIBUTES

# %%
p = {'max_depth': 8, 'eta': 0.1, 'gamma': 1, 'subsample': 0.2, 'lambda': 0.1, 
     'colsample_bytree': 0.6, 'scale_pos_weight': 0.6097560975609756, 'seed': 1618, 
     'nthread': 4, 'objective': 'binary:logistic'}


train = pd.read_csv(snakemake.input.train, sep='\t')
val = pd.read_csv(snakemake.input.validation, sep='\t')

train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

# SCALING
ss = StandardScaler()
train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)


train_dmat = xgboost.DMatrix(train_X_s, train_y)
val_dmat = xgboost.DMatrix(val_X_s, val_y)

# %%
m = xgboost.train(p, train_dmat, num_boost_round=100, early_stopping_rounds=15,
                  evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

# %%
m.save_model(snakemake.output.xgb_final)