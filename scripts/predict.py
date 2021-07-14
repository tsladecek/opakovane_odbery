#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
from scripts.constants import ATTRIBUTES


# %%
def predict(df, train_data="data/train.tsv", val_data="data/validation.tsv",
            model_path="results/models/xgboost_final.json", proba=False):
    
    train = pd.read_csv(train_data, sep='\t')
    val = pd.read_csv(val_data, sep='\t')
    
    train_X = train.loc[:, ATTRIBUTES]
    val_X = val.loc[:, ATTRIBUTES]
    X = df.loc[:, ATTRIBUTES]
    
    # SCALING
    ss = StandardScaler()
    train_val_X_s = ss.fit(pd.concat([train_X, val_X]))
    X_s = xgboost.DMatrix(ss.transform(X), feature_names=ATTRIBUTES)
    
    # predict
    xgb = xgboost.Booster()
    xgb.load_model(model_path)
    
    yhat = xgb.predict(X_s)
    
    # return xgb
    if not proba:
        return (yhat > 0.5) * 1
    
    return yhat

# %%
import numpy as np
from sklearn.metrics import confusion_matrix

test = pd.read_csv("data/test.tsv", sep='\t')

yhat_test = predict(test)

(TN, FP), (FN, TP) = confusion_matrix(test.binary_2nd_result.values, yhat_test)
test_acc = (TN + TP) / (TN + TP + FP + FN)  # == 58.6 %
test_sens = TP / (TP + FN)  # 68.75 %
test_spec = TN / (TN + FP)  # 49.15 %
test_mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) # 0.1447

