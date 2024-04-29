# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:07:03 2023

@author: Heng2020
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
# joblib: for save&load pipeline
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import time
import csv
import seaborn as sns
from playsound import playsound
import myLib01.ungroupped01 as f

sound_path = r"H:\D_Music\Women Laugh.wav"


def _xgb_RMSE_H2(X_train, y_train,single_param,cv=10):
    # _xgb_RMSE_H1 helps convert cat columns
    # not tested
    steps = [("ohe_onestep", DictVectorizer(sparse=False)),
              ("xgb_model", xgb.XGBClassifier(**single_param))]
    xgb_pipe = Pipeline(steps)
    # cross_val_scores has no random state 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    # bottle neck is below
    cross_val_scores = cross_val_score(
                            xgb_pipe,
                            X_train.to_dict("records"),
                            y_train,
                            scoring = "accuracy",
                            cv=cv,
                            )
    # bottle neck is below
    xgb_pipe.fit(X_train.to_dict("records"),y_train)
    
    RMSE_cross = cross_val_scores.mean()

    return RMSE_cross

df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Classification/Classify 02/02 Skin Disease.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "rt"
saved_model_name = "Model 01.01"
mySeed = 20
num_to_cat_col = []
# n_data = (string means used all data) || eg. n_data = "all"
n_data = "all"


folder_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 06"
null_report = f.count_null(data)
saved_model_path = folder_path + "/" + saved_model_name + ".joblib"

data = f.num_to_cat(data,num_to_cat_col)

if isinstance(n_data, str):
    data_used = data
else:
    data_used = data.sample(n=n_data,random_state=mySeed)

X_train_full,X_dev_full, X_test_full, y_train,y_dev, y_test = f.train_dev_test_split(
                                                    data_used.drop(y_name, axis=1),
                                                    data_used[y_name],
                                                    seed=mySeed
                                                    )
X_train = X_train_full
X_dev = X_dev_full
X_test = X_test_full

param_01 = {
    'objective':'multi:softmax',
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': 0.2,
    # n_estimators = 330 is best(so far)
    # n_estimators range is 300-400
    #'n_estimators': np.arange(250, 400, 20), 
    'max_depth': 4,
    'subsample': 0.7
    }

#################### input

single_param = param_01
cv = 10
#------------------ input

steps = [("ohe_onestep", DictVectorizer(sparse=False)),
          ("xgb_model", xgb.XGBClassifier(**single_param))]
xgb_pipe = Pipeline(steps)
# cross_val_scores has no random state 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
# bottle neck is below
cross_val_scores = cross_val_score(
                        xgb_pipe,
                        X_train.to_dict("records"),
                        y_train,
                        scoring = "accuracy",
                        cv=cv,
                        )
# bottle neck is below
xgb_pipe.fit(X_train.to_dict("records"),y_train)

RMSE_cross = cross_val_scores.mean()



