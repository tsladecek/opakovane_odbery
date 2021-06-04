#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation of Gridsearch Results 
"""

# %%
import pandas as pd

# %%
res = []

for m in ["logisticregression", "lda", "qda", "svc", "randomforest", "xgboost"]:
    df = pd.read_csv(f"results/gridsearch_results/{m}.tsv", sep='\t', index_col=0)
    res.append([m] + df.iloc[0].values.tolist())
    
results = pd.DataFrame(res, columns = ['model'] + list(df))
results.to_csv("results/summary.tsv", sep='\t', index=False)
