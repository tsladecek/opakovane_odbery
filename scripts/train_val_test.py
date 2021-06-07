#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Validation Test data
"""

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("data/uninformatives.tsv", sep='\t')

# %%
train, valtest = train_test_split(df, test_size=0.3, random_state=1618)
validation, test = train_test_split(valtest, test_size=0.5, random_state=1618)

# %%
train.to_csv('data/train.tsv', sep='\t', index=False)
validation.to_csv('data/validation.tsv', sep='\t', index=False)
test.to_csv('data/test.tsv', sep='\t', index=False)