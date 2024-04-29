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
from sklearn.model_selection import KFold
# StratifiedKFold is used to ensure that class distribution is maintained across folds.
from sklearn.model_selection import StratifiedKFold

df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 04_GLM\04 Credit Risk Customer.csv"
y_name = "class"
saved_model_name = "Model LGB.01"

folder_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 04_GLM"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"


drop_col01 = []
drop_col02 = []
drop_col03 = []
drop_col = drop_col01


data = pd.read_csv(df_path,header=0)

mySeed = 20
num_to_cat_col = []



saved_model_path = folder_path + "/" + saved_model_name + ".joblib"
data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data):

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



train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test, label=y_test)

params01 = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    # auto numclass
    'num_class': 2,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    # 'n_estimators': 100,
    'max_depth':8
}


model = lgb.LGBMClassifier(loss_function= 'Logloss', custom_metric=['Accuracy','AUC'],eval_metric='F1')
model.fit(X_train, y_train, eval_set=(X_test, y_test), feature_name='auto', categorical_feature = 'auto',verbose=0)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=mySeed)

results = []
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=mySeed)

temp = skf.split(X_train,y_train)

for train_index, test_index in skf.split(data):
    # Split the data into training and testing sets for each fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training set
    model = sm.Logit(y_train, sm.add_constant(X_train))
    result = model.fit()

    # Evaluate the model on the testing set
    y_pred = result.predict(sm.add_constant(X_test))
    accuracy = np.mean((y_pred >= 0.5) == y_test)

    # Store the results
    results.append(accuracy)

cv_results = lgb.cv(params01, train_data,folds=skf.split(X_train, y_train))

y_pred_train = model.predict(X_train)

print(metrics.classification_report(y_train, y_pred_train))
print(metrics.confusion_matrix(y_train, y_pred_train))

feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))

y_pred_test = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred_test))
print(metrics.confusion_matrix(y_test, y_pred_test))
