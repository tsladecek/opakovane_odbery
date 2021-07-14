#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snakemake Input Functions
"""

def all_models(wildcards):
    res = []
    for m in ["logisticregression", "lda", "qda", "svc", "randomforest", "xgboost"]:
        res.append(f"results/models/{m}.json")
        res.append(f"results/gridsearch_results/{m}.tsv")
    res.append("results/models/xgboost_final.json")    
    return res

