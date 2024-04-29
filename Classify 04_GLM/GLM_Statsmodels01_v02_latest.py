# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:10:07 2023

@author: Heng2020
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import myLib01.ungroupped01 as f01
import myLib01.ungroupped02 as f02

import myLib01.lib02_dataframe as l02

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import pickle
# still can't play alarm
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# Model1 train_accuracy = 78%(without cross-validation)
# Model1 test_accuracy = 73%(without cross-validation)


# alarm_path = r"C:/Users/n1603499/Music/Sound Effect positive-logo-opener.wav"

# alarm_path02 = r"C:\Users\n1603499\Music\Sound Effect positive-logo-opener.wav"

# alarm_path03 = r"C:\Users\n1603499\OneDrive - Liberty Mutual\Documents\Sound Effect positive-logo-opener.wav"
# alarm = AudioSegment.from_wav(alarm_path03)
# play(alarm)
# playsound(alarm_path02)

# df_path = r"C:\Users\n1603499\OneDrive - Liberty Mutual\Documents\15.01 PDM Demand\Project 01_SG Retention Model\Dummy\04 Credit Risk Customer.csv"

df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Classification/Classify 04_GLM/04 Credit Risk Customer.csv"
data = pd.read_csv(df_path,header=0)
# print(data.head(5))
y_name = "class"
saved_model_name = "Model 02"
mySeed = 20
num_to_cat_col = ["existing_credits","num_dependents"]

drop01 = ["personal_status"]
drop02 = ["personal_status","own_telephone"]

drop_col = drop02

label_encode_name = "Label01"
folder_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 04_GLM"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"

saved_model_path = folder_path  + "/"  + saved_model_name + ".pickle"
saved_label_path = folder_path  + "/"  + label_encode_name + ".pickle"

used_cat_col = []
used_num_col = []

n_cv = 5

######################## Class ########################
class Stat_Model():
    def __init__(self):
        pass
    def add_num_column(self,num_column):
        self.num_column = num_column
        
    def add_cat_column(self,cat_column):
        self.cat_column = cat_column
    
    def add_model(self,model):
        self.model_obj = model

class Preprocessing(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    
    def transform(self,X):
        return X

class NumToCat(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    
    def transform(self,X):
        tab01 = ["color","art"]
        return f01.num_to_cat(X,tab01)

class FeatureEncoding(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    
    def transform(self,X,y_name):
        data_dummy = l02.create_dummy(X,y_name)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(data_dummy[y_name])
        data_dummy[y_name] = encoded_labels
        return data_dummy
    
################################### Pre processing  ##################

def pd_preprocess(data):
    return data
########################### Other special functions ##########################
def sm_predit_prob(model,data,inplace=True,label_encoder=None):
    # label_encoder can be LabelEncoder object or path
    feature_used = sm_feature_name(model)
    data_only_needed_X = data[feature_used]
    
    y_pred_name = model02.model.endog_names
    data_dummy = f02.create_dummy(data,y_pred_name)
    
    encoded_labels = label_encoder.fit_transform(data_dummy[y_name])
    data_dummy[y_name] = encoded_labels
    
    y_pred_prob = model.predict(data)
    if inplace:
        out_df = pd.concat([data,y_pred_prob],axis=1)
        return out_df
    else:
        return y_pred_prob
    

def count_prefix(list_in, prefix_list,return_as_list=True):
    # return_as_list=False will return dictionary
    if isinstance(prefix_list, str):
        ans = sum([1 for element in list_in if element.startswith(prefix_list)])
        return ans
    else:
        ans_list = []
        for prefix in prefix_list:
            temp = sum([1 for element in list_in if element.startswith(prefix)])
            ans_list.append(temp)
            
        if return_as_list:
            return ans_list
        else:
            out_dict = dict()
            for i,n in enumerate(ans_list):
                out_dict[prefix_list[i]] = n
                
            return out_dict

def _cat_or_num_col(col_name,sm_col_list):
    n_count = count_prefix(sm_col_list, col_name)
    if n_count > 1:
        return "cat"
    else:
        return "num"
    


def sm_predit_label():
    pass


def _sm_feature_nameH1(lst):
    # what about numberic columns?
    unique_values = set()
    extracted_values = []
    
    for value in lst:
        if value not in ["const"]:
            
            extracted_value = value.rsplit('_', 1)[0]
            
            if _cat_or_num_col(extracted_value, lst) == "cat" :
                
                extracted_values.append(extracted_value)
                unique_values.add(extracted_value)
            else:
                extracted_values.append(value)
                unique_values.add(value)

    return list(unique_values)

def sm_feature_name(model):
    feature_list = model.params.index.tolist()
    ans = _sm_feature_nameH1(feature_list)
    return ans


#-------------------------- Other special functions --------------------------

############################## Plotting Charts Function #####################################
def lift_chart(actual, pred_prob, title, bins=10):
    df = pd.DataFrame()
    df['actual'] = actual
    df['predicted'] = pred_prob
    df['quantile'] = pd.qcut(df['predicted'], bins, labels = False,duplicates="drop") 
    chart = df.groupby('quantile').agg({'actual':'mean', 'predicted':'mean'})

    _ = chart[['actual' , 'predicted']].plot.line(title=title)
    plt.ylim(0)
    y_pred_num = (y_pred_train_prob > 0.5).astype(int)
    accuracy = accuracy_score(actual, y_pred_num)

    print(f'{title} accuarcy: {accuracy:0.5f}')

def roc_plot(actual_label,pred_prob):
    auc = roc_auc_score(actual_label,pred_prob)
    fpr,tpr,_ = roc_curve(y_train,pred_prob)
    
    
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylim(0)
    plt.ylabel('True Positive Rate')
    plt.title('Train ROC Curve (AUC = {:.2f})'.format(auc))
    plt.show()

#---------------------------- Plotting Charts Function  --------------------------------

data_full = data.copy()
data = data.drop(drop_col,axis=1)
data = pd_preprocess(data)

data = f01.num_to_cat(data,num_to_cat_col)

used_num_col = l02.df_num_column(data)
used_cat_col = l02.df_cat_column(data)


Model01 = Stat_Model()
Model01.add_num_column(used_num_col)
Model01.add_cat_column(used_cat_col)



data_dummy = l02.create_dummy(data,y_name)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data_dummy[y_name])
data_dummy[y_name] = encoded_labels

data_full_dummy = l02.create_dummy(data_full,y_name)

X_train, X_test, y_train, y_test = train_test_split(
                                        data_dummy.drop(y_name, axis=1), 
                                        data_dummy[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)

pipe01 = Preprocessing()
pipe02 = NumToCat()
pipe03 = FeatureEncoding()

pipe02.num_to_cat_col = num_to_cat_col
data_transformed = pipe02.fit_transform( pipe01.fit_transform(data),num_to_cat_col)





# get the result back
# decoded_labels = label_encoder.inverse_transform(df['class'])
# df['class_decoded'] = decoded_labels

model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
model_fit = model.fit()
print(model_fit.summary())

Model01.add_model(model)
f02.obj_function(Model01)
f02.obj_property(Model01)

y_pred_train_prob = model_fit.predict(X_train)


y_pred_train_num = (y_pred_train_prob > 0.5).astype(int)
y_pred_train = label_encoder.inverse_transform(y_pred_train_num)

print(metrics.classification_report(y_train, y_pred_train_num))
print(metrics.confusion_matrix(y_train, y_pred_train_num))



y_pred_test_prob = model_fit.predict(X_test)
y_pred_test_num = (y_pred_test_prob > 0.5).astype(int)

y_pred_test = label_encoder.inverse_transform(y_pred_test_num)
print(metrics.classification_report(y_test, y_pred_test_num))
print(metrics.confusion_matrix(y_test, y_pred_test_num))

with open(saved_model_path, 'wb') as file:
    pickle.dump(model, file)

with open(saved_model_path, 'rb') as file:
    model02 = pickle.load(file)
#################### save label encoding
with open(saved_label_path, 'wb') as file:
    pickle.dump(label_encoder, file)

with open(saved_label_path, 'rb') as file:
    label01 = pickle.load(file)
    
    
#########################  Additional gragh to analyse model result

roc_plot(y_train, y_pred_train_prob)
lift_chart(y_train, y_pred_train_prob, "Train")

# ans01 = model02.predict(data_full_dummy)
# pred_prob01 = sm_predit_prob(model02, data,True,label01)
ans02 = sm_feature_name(model02)



