
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:48:33 2023

@author: Heng2020
"""
# Next is add cross-validation
# and tuning param framework
# v02 => LightGBM

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

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\02 DataFrame")
import lib02_dataframe as ds



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

data = ds.num_to_cat(data,num_to_cat_col)
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
    'max_depth':8,
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

callbacks = [lgb.log_evaluation(period=50)]

model = lgb.LGBMClassifier()
model.fit(X_train, y_train, eval_set=(X_test, y_test),callbacks=callbacks)

# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=mySeed)

# results = []
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=mySeed)

# temp = skf.split(X_train,y_train)

# for train_index, test_index in skf.split(data):
#     # Split the data into training and testing sets for each fold
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     # Fit the model on the training set
#     model = sm.Logit(y_train, sm.add_constant(X_train))
#     result = model.fit()

#     # Evaluate the model on the testing set
#     y_pred = result.predict(sm.add_constant(X_test))
#     accuracy = np.mean((y_pred >= 0.5) == y_test)

#     # Store the results
#     results.append(accuracy)


cv_results = lgb.cv(params01, train_data,nfold=num_folds, stratified=True)

y_pred_train = model.predict(X_train)

print(metrics.classification_report(y_train, y_pred_train))
print(metrics.confusion_matrix(y_train, y_pred_train))

feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))

y_pred_test = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred_test))
print(metrics.confusion_matrix(y_test, y_pred_test))
