# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:15:17 2024

@author: Heng2020
"""

import seaborn as sns
import matplotlib.pyplot as plt


import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
import lib02_dataframe as f 


df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\07 recipe_site_traffic_2212.csv"
y_name = "high_traffic"
saved_model_name = "Model LGB.01"

folder_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 04_GLM"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"


drop_col01 = []
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

count_nan =  data['calories'].isna().sum()
n_rows = data_ori.shape[0]


def display_pie(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d} sec )"

t_list = [t_02_import_policy ,t_03_import_claim,t_04_policy,t_05_claim,t_06_polclm,t_07_parquet ,t_08_end_sample]
labels = [ "t_02_import_policy ", "t_03_import_claim", "t_04_policy", "t_05_claim", "t_06_polclm", "t_07_parquet ", "t_08_sample "]
