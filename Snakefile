configfile: "config.yaml"
include: "scripts/input_functions.py"

import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.gridsearch import gridsearch
from scripts.model_search_space import model_search_space
from scripts.constants import ATTRIBUTES


rule all_models:
    input:
        all_models



rule model_gridsearch:
    input:
        train      = "data/train.tsv",
        validation = "data/validation.tsv"
    output:
        modelpath  = "results/models/{model}.json",
        results    = "results/gridsearch_results/{model}.tsv"
    threads: config["THREADS"]
    run:
        train = pd.read_csv(input.train, sep='\t', index_col=0)
        val = pd.read_csv(input.validation, sep='\t', index_col=0)
        
        train_X = train.loc[:, ATTRIBUTES]
        train_y = train.binary_2nd_result.values
        
        val_X = val.loc[:, ATTRIBUTES]
        val_y = val.binary_2nd_result.values
        
        # SCALING
        ss = StandardScaler()
        train_X_s = ss.fit_transform(train_X)
        val_X_s = ss.transform(val_X)
       
        # GRIDSEARCH 
        gridsearch(train_X, train_y, val_X, val_y, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)
