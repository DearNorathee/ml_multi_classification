# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:26:15 2024

@author: Heng2020
"""

import lightgbm as lgb

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
import lib02_dataframe as f 

def cat_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    out_dict = {}
    out_dict['accuracy'] = accuracy_score(y_true,y_pred)
    out_dict['f1'] = f1_score(y_true,y_pred)
    out_dict['precision'] = precision_score(y_true,y_pred)
    out_dict['recall'] = recall_score(y_true,y_pred)
    
    return out_dict

df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\07 recipe_site_traffic_2212.csv"
y_name = "high_traffic"
saved_model_name = "Model LGB.01"

positive_class = "High"
eval_metric='precision'
extra_metrics = ['f1','precision','recall','accuracy']

folder_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 04_GLM"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"


drop_col01 = ['recipe']
drop_col02 = []
drop_col03 = []
drop_col = drop_col01


data_ori = pd.read_csv(df_path,header=0)

mySeed = 20
num_to_cat_col = []

n_data = 10_000

if isinstance(n_data, str) or data_ori.shape[0] < n_data :
    data = data_ori
else:
    data = data_ori.sample(n=n_data,random_state=mySeed)


saved_model_path = folder_path + "/" + saved_model_name + ".joblib"
data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data):
    df_cleaned = data.dropna(subset=['calories'])
    df_cleaned['high_traffic'] = df_cleaned['high_traffic'].astype('object')
    df_cleaned['high_traffic'] = df_cleaned['high_traffic'].fillna('Low')
    
    df_cleaned['servings'] = df_cleaned['servings'].replace({'4 as a snack': 4, '6 as a snack': 6})
    df_cleaned['servings'] = df_cleaned['servings'].astype(int)
    df_cleaned['category'] = data['category'].astype('category')
    return df_cleaned


#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
data = pd_preprocess(data)

data = f.pd_num_to_cat(data,num_to_cat_col)
cat_col = f.pd_cat_column(data)
data = f.pd_to_category(data)

# convert positive class to 1, negative class to 0
# as of now autogluon, doesn't have a way to specify 
# autogluon version 0.8.2, Jan 6, 2024
data[y_name] = np.where(data[y_name] == positive_class,1,0 )

X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(y_name, axis=1), 
                                        data[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)

train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test, label=y_test)

train_df = pd.concat([X_train,y_train],axis=1)
test_df = pd.concat([X_test,y_test],axis=1)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)


predictions = model.predict(X_train)
train_prediction = train_df.copy()
train_prediction[y_name + "_predict"] = predictions
train_metric = cat_metrics(train_prediction[y_name], train_prediction[y_name + "_predict"])
train_metric

predictions = model.predict(X_test)
test_prediction = test_df.copy()
test_prediction[y_name + "_predict"] = predictions
test_metric = cat_metrics(test_prediction[y_name], test_prediction[y_name + "_predict"])
test_metric
