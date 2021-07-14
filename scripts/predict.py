#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
from scripts.constants import ATTRIBUTES


# %%
def predict(df, train_data="data/train.tsv", model_path="results/models/xgboost_final.json", proba=False):
    train = pd.read_csv(train_data, sep='\t')
    
    train_X = train.loc[:, ATTRIBUTES]
    
    X = df.loc[:, ATTRIBUTES]
    
    # SCALING
    ss = StandardScaler()
    train_X_s = ss.fit_transform(train_X)
    X_s = xgboost.DMatrix(ss.transform(X), feature_names=ATTRIBUTES)
    
    # predict
    xgb = xgboost.Booster()
    xgb.load_model(model_path)
    
    yhat = xgb.predict(X_s)
    
    # return xgb
    if not proba:
        return (yhat > 0.5) * 1
    
    return yhat