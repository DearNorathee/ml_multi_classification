# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:13:29 2023

@author: Heng2020
"""


import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

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


train_df = pd.concat([X_train,y_train],axis=1)
test_df = pd.concat([X_test,y_test],axis=1)
# Load data
train_data = TabularDataset(train_df) 


# Initialize predictor # train using precision
predictor = TabularPredictor(label=y_name, eval_metric = eval_metric).fit(train_data)
results = predictor.fit_summary(show_plot=True)
results_train = predictor.leaderboard(train_df, silent=True,extra_metrics=extra_metrics)
results_test = predictor.leaderboard(test_df, silent=True,extra_metrics=extra_metrics)

#### prediction on training data
predictions = predictor.predict(train_df)
train_prediction = train_df.copy()
train_prediction[y_name + "_predict"] = predictions
precision_train_metric = cat_metrics(train_prediction[y_name], train_prediction[y_name + "_predict"])


# Making predictions on new data
predictions = predictor.predict(test_df)
test_prediction = test_df.copy()
test_prediction[y_name + "_predict"] = predictions

permutation_importances = predictor.feature_importance(data=train_data,model = predictor.get_model_names())


best_model_name = predictor.get_model_best()
best_model_params = predictor.fit_hyperparameters_
print(best_model_params)
######### there are only 2 positives
################# train on f1
# Initialize predictor
# .fit(train_data,num_bag_folds=5): cross-validation
predictor_f1 = TabularPredictor(label=y_name, eval_metric = 'f1').fit(train_data,num_bag_folds=5)
results_f1 = predictor_f1.fit_summary(show_plot=True)
results_f1_train = predictor_f1.leaderboard(train_df, silent=True,extra_metrics=extra_metrics)
results_f1_test = predictor_f1.leaderboard(test_df, silent=True,extra_metrics=extra_metrics)

predictions_f1 = predictor_f1.predict(train_df)
train_prediction_f1 = train_df.copy()
train_prediction_f1[y_name + "_predict"] = predictions_f1
f1_train_metric = cat_metrics(train_prediction_f1[y_name], train_prediction_f1[y_name + "_predict"])

# Making predictions on new data
predictions_f1 = predictor_f1.predict(test_df)
test_prediction_f1 = test_df.copy()
test_prediction_f1[y_name + "_predict"] = predictions_f1
f1_test_metric = cat_metrics(test_prediction_f1[y_name], test_prediction_f1[y_name + "_predict"])


permutation_importances = predictor.feature_importance(data=train_data,model = predictor.get_model_names())


################# train on accuracy
# Initialize predictor
predictor_acc = TabularPredictor(label=y_name, eval_metric = 'accuracy').fit(train_data)
results_acc = predictor.fit_summary(show_plot=True)
results_acc_train = predictor.leaderboard(train_df, silent=True,extra_metrics=extra_metrics)
results_acc_test = predictor.leaderboard(test_df, silent=True,extra_metrics=extra_metrics)

