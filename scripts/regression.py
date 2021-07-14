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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

from scripts.constants import ATTRIBUTES

from scripts.gridsearch import gridsearch 
from scripts.model_search_space import model_search_space

from matplotlib import rcParams


rcParams.update({"font.size": 15})

# %%
# train = pd.read_csv("data/train.tsv", sep='\t')
# val = pd.read_csv("data/validation.tsv", sep='\t')
# test = pd.read_csv("data/test.tsv", sep='\t')

train = pd.read_csv(snakemake.input.train, sep='\t')
val = pd.read_csv(snakemake.input.validation, sep='\t')
test = pd.read_csv(snakemake.input.test, sep='\t')

train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

test_X = test.loc[:, ATTRIBUTES]
test_y = test.binary_2nd_result.values

all_X = pd.concat([train, val, test])

# SCALING
ss = StandardScaler()


train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)
test_X_s = ss.transform(test_X)


# # %% Regression
poly = PolynomialFeatures(degree=2)
train_poly = poly.fit_transform(train_X)

poly = PolynomialFeatures(degree=2)
val_poly = poly.fit_transform(val_X)

train_X_poly = ss.fit_transform(train_poly)
val_X_poly = ss.transform(val_poly)

y_train = train.second_ff.values
y_val = val.second_ff.values
y_test = test.second_ff.values

# %%
def trymodel(model, params, X, y, X_val, y_val):
    model.set_params(**params)
    
    model.fit(X, y)
    
    yhat_train = model.predict(X)
    yhat_val = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y, yhat_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, yhat_val))
    
    # print(f'Train RMSE     : {train_rmse}')
    # print(f'Validation RMSE: {val_rmse}')
    
    return train_rmse, val_rmse, model


# %% Ridge
# ridge_res = []
# for alpha in np.linspace(20, 80, 41):
#     t, v, m = trymodel(Ridge(), {"alpha": alpha}, train_X_s, y_train, val_X_s, y_val)
    
#     ridge_res.append([alpha, t, v])
    
# ridge_res = np.array(ridge_res)

# plt.plot(ridge_res[:, 0], ridge_res[:, 1], '.', label='train')
# plt.plot(ridge_res[:, 0], ridge_res[:, 2], '.', label='validation')
# plt.legend()


# %%
ALPHA = 50

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 12))

sns.histplot(all_X.loc[:, "Gestational age"], bins=10, color='lightgrey', ax=ax[0, 0])
ax[0, 0].set_xlabel("Gestational age (in days)")
ax[0, 0].set_title("A", loc="left", fontsize=30)


sns.lineplot(x = "Gestational age", y = "second_ff", style="binary_2nd_result", 
             data = all_X, ax=ax[0, 1], color='black')
ax[0, 1].set_title("B", loc="left", fontsize=30)

sns.regplot(x = "Gestational age", y = "binary_2nd_result", color='black', data=all_X,
             ax=ax[1, 0], lowess=True)
ax[1, 0].set_title("C", loc="left", fontsize=30)

t, v, m = trymodel(Ridge(), {"alpha": ALPHA}, train_X_s, y_train, val_X_s, y_val)

yhat_test = m.predict(test_X_s)
ax[1, 1].plot(y_test, yhat_test, '.k', markersize=20)
ax[1, 1].plot(np.linspace(0, 0.13, 10), np.linspace(0, 0.13, 10), '--k', alpha=0.2)

ax[1, 1].set_xlim(0.02, 0.13)
ax[1, 1].set_ylim(0.02, 0.13)

ax[1, 1].set_xlabel('Fetal Fraction after second sampling')
ax[1, 1].set_ylabel('Predicted Fetal Fraction after second sampling')

ax[1, 1].set_title("D", loc="left", fontsize=30)

plt.tight_layout()

plt.savefig(snakemake.output.fourplots, dpi=200)
# plt.savefig('plots/4plots.png', dpi=200)

# %%
fig, ax = plt.subplots(figsize=(9, 7))

t, v, m = trymodel(Ridge(), {"alpha": ALPHA}, train_X_s, y_train, val_X_s, y_val)
t, te, m = trymodel(Ridge(), {"alpha": ALPHA}, train_X_s, y_train, test_X_s, y_test)

yhat_train = m.predict(train_X_s)
yhat_val = m.predict(val_X_s)
ax.plot(y_test, yhat_test, '.k', markersize=20)
ax.plot(np.linspace(0, 0.13, 10), np.linspace(0, 0.13, 10), '--k', alpha=0.2)

ax.set_xlim(0.02, 0.13)
ax.set_ylim(0.02, 0.13)

def baseline_model(n):
    return np.repeat(np.mean(y_train), n)

baseline_val = np.sqrt(mean_squared_error(y_val, baseline_model(len(y_val))))  # np.sqrt(np.mean((y_val - np.mean(y_train)) ** 2))
baseline_train = np.sqrt(mean_squared_error(y_train, baseline_model(len(y_train))))  # np.sqrt(np.mean((y_train - np.mean(y_train)) ** 2))
baseline_test = np.sqrt(mean_squared_error(y_test, baseline_model(len(y_test))))

plt.title('Ridge (Test Data)\
          \nRMSE - Train: {:.3f}, Validation: {:.3f}, Test: {:.3f}\
          \nBaseline RMSE - Train: {:.3f}, Validation: {:.3f}, Test: {:.3f}'\
              .format(t, v, te, baseline_train, baseline_val, baseline_test))
# ax.set_title('Train RMSE: {:.3f}, Validation RMSE: {:.3f}'.format(t, v))

ax.set_xlabel('Fetal Fraction after second sampling')
ax.set_ylabel('Predicted Fetal Fraction after second sampling')

plt.tight_layout()

plt.savefig(snakemake.output.ridge, dpi=200)
# plt.savefig('plots/ridge.png', dpi=100)

# %%
# print(np.corrcoef(y_val, yhat_val))

# # %% Ridge with polynomial features - Poorer performance that normal ridge
# ridge_res = []
# for alpha in np.linspace(0, 1000, 41):
#     t, v, m = trymodel(Ridge(), {"alpha": alpha}, train_X_poly, y_train, val_X_poly, y_val)
    
#     ridge_res.append([alpha, t, v])
    
# ridge_res = np.array(ridge_res)

# plt.plot(ridge_res[:, 0], ridge_res[:, 1], '.', label='train')
# plt.plot(ridge_res[:, 0], ridge_res[:, 2], '.', label='validation')
# plt.legend()

# %% SVR - UNUSABLE
# svr_res = []
# for c in np.linspace(0.0001, 0.001, 41):
#     t, v, m = trymodel(SVR(), {"kernel": "rbf", 'gamma': 1e-2, 'C': c}, train_X_s, y_train, val_X_s, y_val)
    
#     svr_res.append([c, t, v])
    
# svr_res = np.array(svr_res)

# plt.plot(svr_res[:, 0], svr_res[:, 1], '.', label='train')
# plt.plot(svr_res[:, 0], svr_res[:, 2], '.', label='validation')
# plt.legend()

# %% xgboost
# xgb_res = []
# for lmbd in np.linspace(0, 1, 41):
#     t, v, m = trymodel(XGBRegressor(),
#                        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.2,
#                         "reg_lambda": lmbd, "subsample": 0.3},
#                        train_X, y_train, val_X, y_val)
    
#     xgb_res.append([lmbd, t, v])
    
# xgb_res = np.array(xgb_res)

# plt.plot(xgb_res[:, 0], xgb_res[:, 1], '.', label='train')
# plt.plot(xgb_res[:, 0], xgb_res[:, 2], '.', label='validation')
# plt.legend()

# # %%
# yhat_val = m.predict(val_X)
# plt.plot(y_val, yhat_val, '.k')
# plt.plot(np.linspace(0, 0.1, 10), np.linspace(0, 0.1, 10), '--k', alpha=0.2)

# plt.xlim(0.02, 0.1)
# plt.ylim(0.02, 0.1)


# plt.title('Validation Performance - Ridge\nTrain RMSE: {:.3f}, Validation RMSE: {:.3f}'.format(t, v) )
# plt.xlabel('Real Second FF')
# plt.ylabel('Predicted Second FF')