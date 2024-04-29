# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:00:20 2024

@author: Heng2020
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:00:20 2024

@author: Heng2020
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
# StratifiedKFold is used to ensure that class distribution is maintained across folds.
from sklearn.model_selection import StratifiedKFold
import warnings


from AutoHyperTuner import AutoHyperTuner



import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\02 DataFrame")
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\03 Modeling")
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\01 XgBoost")

import lib01_xgb as xgt
import lib02_dataframe as ds
import lib03_modeling as ml




def test_AutoHyperTuner():




    df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\08 ObesityRisk\08 ObesityRisk_train.csv"
    y_name = "NObeyesdad"
    saved_model_name = "LightGBM Obesity_risk_v01"

    # positive_class = "High"
    eval_metric = 'accuracy'


    folder_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Classification/Classify 08"
    model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
    alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"

    drop_col01 = ['id']
    drop_col02 = ['id']
    drop_col03 = []
    drop_col = drop_col01


    data_ori = pd.read_csv(df_path,header=0)

    mySeed = 20
    num_to_cat_col = []

    n_data = 30_000

    if isinstance(n_data, str) or data_ori.shape[0] < n_data :
        data = data_ori
    else:
        data = data_ori.sample(n=n_data,random_state=mySeed)


    saved_model_path = folder_path + "/" + saved_model_name
    data = data.drop(drop_col,axis=1)

    ################################### Pre processing - specific to this data(Blood Pressure) ##################
    def pd_preprocess(data):
        df_cleaned = data

        return df_cleaned

    null_report = ds.count_null(data)



    #---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
    data = pd_preprocess(data)

    data = ds.pd_num_to_cat(data,num_to_cat_col)
    cat_col = ds.pd_cat_column(data)
    data = ds.pd_to_category(data)

    X_train, X_test, y_train, y_test = train_test_split(
                                            data.drop(y_name, axis=1), 
                                            data[y_name], 
                                            test_size=0.2, 
                                            random_state=mySeed)



    train_data = lgb.Dataset(data=X_train, label=y_train)
    test_data = lgb.Dataset(data=X_test, label=y_test)

    params01 = {
        # 'objective': 'multiclass',
        # 'metric': 'multi_logloss',
        # auto numclass
        # 'num_class': 2,
        'max_depth':np.arange(2,20,1),
        'num_leaves': 31,
        'max_bin': 260,
        
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        
        
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'bagging_freq': 5,
        'verbosity': -1,
        # 'n_estimators': 100,
        # 'num_boost_round' : 1000,
        
        "min_data_in_leaf":100
    }

    params02 = {
        # 'objective': 'multiclass',
        # 'metric': 'multi_logloss',
        # auto numclass
        # 'num_class': 2,
        'max_depth':15,
        'num_leaves': 31,
        'max_bin': 260,
        
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        
        
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'bagging_freq': 5,
        'verbosity': -1,
        # 'n_estimators': 100,
        # 'num_boost_round' : 1000,
        
        "min_data_in_leaf":100
    }


    # callbacks = [lgb.log_evaluation(period=50)]
    # model.fit(X_train, y_train, eval_set=(X_test, y_test),callbacks=callbacks)
    num_folds = 5
    cv_results = lgb.cv(params02, train_data,nfold=num_folds, stratified=True)
    tuner01 = AutoHyperTuner(train_data)
    tuner01.auto_tune()
    # model = lgb.LGBMClassifier()

def main():
    test_AutoHyperTuner()

if __name__ == "__main__":
    main()