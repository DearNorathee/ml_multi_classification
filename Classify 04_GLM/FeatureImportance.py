# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:09:35 2023

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

df_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Code Classification\Classify 06\06 Sleep Health.csv"
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