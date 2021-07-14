#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation of Gridsearch Results 
"""
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
from scripts.predict import predict
from sklearn.metrics import confusion_matrix

# %%
res = []

paths = {
    "logisticregression": snakemake.input.logisticregression,
    "lda": snakemake.input.lda,
    "qda": snakemake.input.qda,
    "svc": snakemake.input.svc,
    "randomforest": snakemake.input.randomforest
}

for m in ["logisticregression", "lda", "qda", "svc", "randomforest", "xgboost"]:
    
    if m == "xgboost":
        train = pd.read_csv(snakemake.input.train, sep='\t')
        val = pd.read_csv(snakemake.input.validation, sep='\t')
        
        yhat_train = predict(train, train_data=snakemake.input.train, model_path=snakemake.input.xgboost)
        yhat_val = predict(val, train_data=snakemake.input.train, model_path=snakemake.input.xgboost)
        
        (TN, FP), (FN, TP) = confusion_matrix(train.binary_2nd_result.values, yhat_train)
        t_acc = (TN + TP) / (TN + TP + FP + FN)
        t_sens = TP / (TP + FN)
        t_spec = TN / (TN + FP)
        t_mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        (TN, FP), (FN, TP) = confusion_matrix(val.binary_2nd_result.values, yhat_val)
        v_acc = (TN + TP) / (TN + TP + FP + FN)
        v_sens = TP / (TP + FN)
        v_spec = TN / (TN + FP)
        v_mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        
        res.append([m, t_acc, t_sens, t_spec, t_mcc, v_acc, v_sens, v_spec, v_mcc])
    else:
        results_path = paths[m]
        df = pd.read_csv(results_path, sep='\t')
        df = df.reset_index(drop=True)
        res.append([m] + df.iloc[0, 2:].values.tolist())

results = pd.DataFrame(res, columns = ['model'] + list(df)[2:])
results.to_csv(snakemake.output.summary, sep='\t', index=False, decimal=',')
