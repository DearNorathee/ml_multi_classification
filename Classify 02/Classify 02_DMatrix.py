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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

sound_path = r"H:\D_Music\Women Laugh.wav"

df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Classification/Classify 02/02 Skin Disease.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "class"
saved_model_name = "Model 01.01"
mySeed = 20
num_to_cat_col = ["erythema", "scaling", "definite_borders", "itching", 
                  "koebner_phenomenon", "polygonal_papules", "follicular_papules", 
                  "oral_mucosal_involvement", "knee_and_elbow_involvement", 
                  "scalp_involvement", "family_history", "melanin_incontinence", 
                  "eosinophils_infiltrate", "PNL_infiltrate", "fibrosis_papillary_dermis", 
                  "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", 
                  "clubbing_rete_ridges", "elongation_rete_ridges", 
                  "thinning_suprapapillary_epidermis", "spongiform_pustule", 
                  "munro_microabcess", "focal_hypergranulosis", 
                  "disappearance_granular_layer", "vacuolisation_damage_basal_layer", 
                  "spongiosis", "saw_tooth_appearance_retes", "follicular_horn_plug", 
                  "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate", 
                  "band_like_infiltrate"]

# n_data = (string means used all data) || eg. n_data = "all"
n_data = "all"


folder_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 06"
null_report = f.count_null(data)
saved_model_path = folder_path + "/" + saved_model_name + ".json"

data = f.num_to_cat02(data,num_to_cat_col)

if isinstance(n_data, str):
    data_used = data
else:
    data_used = data.sample(n=n_data,random_state=mySeed)

le = LabelEncoder()
data_used[y_name] = le.fit_transform(data_used[y_name])

X_train_full,X_dev_full, X_test_full, y_train,y_dev, y_test = f.train_dev_test_split(
                                                    data_used.drop(y_name, axis=1),
                                                    data_used[y_name],
                                                    seed=mySeed
                                                    )
X_train = X_train_full
X_dev = X_dev_full
X_test = X_test_full

# # decode the labels back to class names
# decoded_labels = le.inverse_transform(encoded_labels)

num_class = data_used[y_name].nunique()
param_01 = {
    'objective':'multi:softmax',
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': 0.2,
    # n_estimators = 330 is best(so far)
    # n_estimators range is 300-400
    'n_estimators': 100, 
    'max_depth': 4,
    'subsample': 0.7,
    'num_class':num_class
    }

###################### create Functions for tuning param ##########################
###################### input ##########################
param_dict = param_dict = {
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': [0.065],
    # n_estimators = 330 is best(so far)
    # n_estimators range is 300-400
    'n_estimators': np.arange(100, 300, 100), 
    'max_depth': [3],
    'subsample': [0.4]
    }

param_01 = param_dict = {
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'objective':'multi:softmax',
    'num_class':num_class,
    'learning_rate': 0.065,
    # n_estimators = 330 is best(so far)
    # n_estimators range is 300-400
    'n_estimators': 100, 
    'max_depth': 3,
    'subsample': 0.4
    }

###################### function inner ##########################
def xgb_Classify_Tune(
        X_train,
        y_train,
        param_dict,
        print_time=True,
        draw_graph=True,
        alarm=True,
        cv=10,
        seed=1,
        num_boost_round=1000,
        early_stopping_rounds=10
        ):
    num_class = y_train.nunique()
    t01 = time.time()
    Xy_DMatrix = xgb.DMatrix(X_train, y_train, enable_categorical=True)   
    
    param_df = f.param_combination(param_dict)
    print(f"Total test: {param_df.shape[0]} ")
    param_test_name = f.tune_param_name(param_dict)
    print(param_test_name)
    
    out_df = param_df.copy()
    
    for i in out_df.index:
        # each_row should be df
        each_row = param_df.loc[[i]].to_dict('records')[0]
        
        each_row['objective'] = 'multi:softmax'
        each_row['num_class'] = num_class
        
        model_cv = xgb.XGBClassifier(**each_row)
        
        # below is the bottle neck
        cv_results = xgb.cv(dtrain=Xy_DMatrix,
                            params=model_cv.get_xgb_params(),
                            nfold=cv,
                            num_boost_round=each_row['n_estimators'],
                            early_stopping_rounds=early_stopping_rounds,
                            metrics="merror",
                            seed=seed,
                            as_pandas=True
                            )
        acc = (1-cv_results["test-merror-mean"]).iloc[-1]
        out_df.loc[i,'Accuracy'] = acc
    
    
    if draw_graph:
        sns.lineplot(x=param_test_name, y='Accuracy',data=out_df)
    if alarm:
        playsound(sound_path)
        
    t02 = time.time()
    t01_02 = t02-t01
    
    if print_time:
        f.output_time(t01_02)
    return out_df


cv_result = xgb_Classify_Tune(X_train,y_train,param_dict)
###################

def xgb_Classify_train(X_train,y_train,param_dict,alarm=True,cv=10,early_stopping_rounds=10):

    Xy_DMatrix = xgb.DMatrix(X_train, y_train, enable_categorical=True)  
    model_out = xgb.train(param_01,Xy_DMatrix)
    
    model_cv = xgb.XGBClassifier(**param_01)
    
    cv_results = xgb.cv(dtrain=Xy_DMatrix,
                        params=model_cv.get_xgb_params(),
                        nfold=10,
                        num_boost_round=param_01['n_estimators'],
                        early_stopping_rounds=early_stopping_rounds,
                        metrics="merror",
                        seed=mySeed,
                        as_pandas=True
                        )
    acc = (1-cv_results["test-merror-mean"]).iloc[-1]
    print(acc)
    return model_out

model01 = xgb_Classify_train(X_train, y_train, param_01)

X_DMatrix = xgb.DMatrix(X_train,enable_categorical=True) 
pred = model01.predict(X_DMatrix)



