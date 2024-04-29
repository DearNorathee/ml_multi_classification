# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:48:33 2023

@author: Heng2020
"""
# Next is add cross-validation
# and tuning param framework
# 

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import myLib01.ungroupped01 as f01
import myLib01.ungroupped02 as f02

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\02 DataFrame")
import lib02_dataframe as ds
from sklearn.metrics import classification_report
from playsound import playsound

df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\08 ObesityRisk\08 ObesityRisk_train.csv"
y_name = "NObeyesdad"
saved_model_name = "AgModel Obesity_risk_v01"

data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "Sleep Disorder"
saved_model_name = "Model 01"
mySeed = 20
num_to_cat_col = []

drop_col01 = ['Person ID']
drop_col02 = ['region']
drop_col03 = []
drop_col = drop_col01



folder_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 06"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"

saved_model_path = folder_path + "/" + saved_model_name + ".joblib"
data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data,col_name = 'Blood Pressure'):
    
    return data


#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
data = pd_preprocess(data)

data = f01.num_to_cat(data,num_to_cat_col)
cat_col = f02.df_cat_column(data)
data = f02.df_to_category(data)

X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(y_name, axis=1), 
                                        data[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)

cat_col = ["Gender","Occupation","BMI Category"]

train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test, label=y_test)

params01 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

model = lgb.LGBMClassifier(loss_function= 'Logloss', custom_metric=['Accuracy','AUC'],eval_metric='F1')
model.fit(X_train, y_train, eval_set=(X_test, y_test), feature_name='auto', categorical_feature = 'auto',verbose=0)

y_pred = model.predict(X_test)


feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
