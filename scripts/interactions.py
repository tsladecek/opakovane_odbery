"""
Test if adding interaction terms will help with the prediction accuracy
"""
import pathlib
# %%
# read train and validation set
import sys

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
from scripts.constants import ATTRIBUTES
import xgboost
import shap

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# train = pd.read_csv('data/train.tsv', sep='\t')
# val = pd.read_csv('data/validation.tsv', sep='\t')
# test = pd.read_csv('data/test.tsv', sep='\t')

train = pd.read_csv(snakemake.input.train, sep='\t')
val = pd.read_csv(snakemake.input.validation, sep='\t')
test = pd.read_csv(snakemake.input.test, sep='\t')

train_X = train.loc[:, ATTRIBUTES]
train_y = train.binary_2nd_result.values

val_X = val.loc[:, ATTRIBUTES]
val_y = val.binary_2nd_result.values

test_X = test.loc[:, ATTRIBUTES]
test_y = test.binary_2nd_result.values

# %% Scale
ss = StandardScaler()
train_X_s = ss.fit_transform(train_X)
val_X_s = ss.transform(val_X)
test_X_s = ss.transform(test_X)

# %% Add interaction terms
poly = PolynomialFeatures(interaction_only=True, include_bias=False)

train_X_scaled_poly = poly.fit_transform(train_X_s)
val_X_scaled_poly = poly.fit_transform(val_X_s)
test_X_scaled_poly = poly.fit_transform(test_X_s)

train_dmat = xgboost.DMatrix(train_X_scaled_poly, train_y)
val_dmat = xgboost.DMatrix(val_X_scaled_poly, val_y)
test_dmat = xgboost.DMatrix(test_X_scaled_poly, test_y)

# %% POLY Attributes

# %% train model
p = {'max_depth': 8, 'eta': 0.1, 'gamma': 1, 'subsample': 0.2, 'lambda': 0.1,
     'colsample_bytree': 0.6, 'scale_pos_weight': 0.6097560975609756, 'seed': 1618,
     'nthread': 4, 'objective': 'binary:logistic'}

m = xgboost.train(p, train_dmat, num_boost_round=100, early_stopping_rounds=15,
                  evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

# metrics
y_hat_val = np.round(m.predict(val_dmat))
(TN, FP), (FN, TP) = confusion_matrix(val_y, y_hat_val)
acc = (TN + TP) / (TN + TP + FP + FN)
sens = TP / (TP + FN)
spec = TN / (TN + FP)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

metrics = pd.DataFrame({
    'accuracy': [acc],
    'sensitivity': [sens],
    'spec': [spec],
    'mcc': [mcc]
})

metrics.to_csv(snakemake.output.poly_train_val, sep='\t', index=False)

# %% predict test
y_hat_test = np.round(m.predict(test_dmat))
(TN, FP), (FN, TP) = confusion_matrix(test_y, y_hat_test)
acc = (TN + TP) / (TN + TP + FP + FN)
sens = TP / (TP + FN)
spec = TN / (TN + FP)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

metrics = pd.DataFrame({
    'accuracy': [acc],
    'sensitivity': [sens],
    'spec': [spec],
    'mcc': [mcc]
})

metrics.to_csv(snakemake.output.poly_test, sep='\t', index=False)

# %% SHAP
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(test_dmat)

shap_values_df = pd.DataFrame(shap_values,
                              columns=poly.get_feature_names_out(input_features=[i.replace(' ', '_') for i in ATTRIBUTES]),
                              dtype=float)
# shap_values_df.to_csv('results/test_interactions_shap_values.tsv', sep='\t')
shap_values_df.to_csv(snakemake.output.shap, sep='\t')

# %% Transform shap values to probability space


# %%
abs_mean_shap_values = [np.mean(np.abs(shap_values[:, i])) for i in range(shap_values.shape[1])]
abs_std_shap_values = [np.std(np.abs(shap_values[:, i])) for i in range(shap_values.shape[1])]

shap_values_summary = pd.DataFrame({
    'attribute': shap_values_df.columns,
    'abs_mean': abs_mean_shap_values,
    'abs_std': abs_std_shap_values
})

# shap_values_df.to_csv('results/test_interactions_shap_values_summary.tsv', sep='\t')
shap_values_df.to_csv(snakemake.output.shap_summary, sep='\t')

# %%
melted = shap_values_df.melt(var_name='attribute', value_name='shap_value')
melted['abs_shap_values'] = np.abs(melted.shap_value)

# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
sns.set_theme(style='ticks', rc={"axes.spines.right": False, "axes.spines.top": False})

# sns.barplot(data=shap_values_summary.sort_values('abs_mean', ascending=False), x="abs_mean", y="attribute", ax=ax, color='grey')
sns.barplot(data=melted, x="abs_shap_values", y="attribute", estimator=np.mean, ax=ax, color='grey')

ax.set_ylabel('')
ax.set_xlabel('Mean of Absolute values of SHAP values')

fig.tight_layout()
# fig.savefig('plots/shap_values_test_interactions.png')
fig.savefig(snakemake.output.shap_plot)
