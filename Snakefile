configfile: "config.yaml"
include: "scripts/input_functions.py"

import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.gridsearch import gridsearch
from scripts.model_search_space import model_search_space
from scripts.constants import ATTRIBUTES

rule all:
    input:
        summary = "results/summary.tsv"



rule summary_table:
    input:
        logisticregression = "results/gridsearch_results/logisticregression.tsv",
        lda = "results/gridsearch_results/lda.tsv",
        qda = "results/gridsearch_results/qda.tsv",
        svc = "results/gridsearch_results/svc.tsv",
        randomforest = "results/gridsearch_results/randomforest.tsv",
        xgboost = "results/models/xgboost_final.json",
        train      = "data/train.tsv",
        validation = "data/validation.tsv"
    output:
        summary = "results/summary.tsv"
    script:
        "scripts/results_evaluation.py"


rule remodel:
    input:
        train      = "data/train.tsv",
        validation = "data/validation.tsv"
    output:
        xgb_final = "results/models/xgboost_final.json"
    script:
        "scripts/remodel.py"

rule model_gridsearch:
    input:
        train      = "data/train.tsv",
        validation = "data/validation.tsv"
    output:
        modelpath  = "results/models/{model}.json",
        results    = "results/gridsearch_results/{model}.tsv"
    threads: config["THREADS"]
    run:
        train = pd.read_csv(input.train, sep='\t')
        val = pd.read_csv(input.validation, sep='\t')
        
        train_X = train.loc[:, ATTRIBUTES]
        train_y = train.binary_2nd_result.values
        
        val_X = val.loc[:, ATTRIBUTES]
        val_y = val.binary_2nd_result.values
        
        # SCALING
        ss = StandardScaler()
        train_X_s = ss.fit_transform(train_X)
        val_X_s = ss.transform(val_X)
       
        # GRIDSEARCH 
        gridsearch(train_X_s, train_y, val_X_s, val_y, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)
