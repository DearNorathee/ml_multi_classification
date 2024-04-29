# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:53:45 2024

@author: Heng2020
"""
# first place is Zenology: 0.92232 as of Feb 17

import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

model_folder = Path(r'C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Classification/Classify 08/')
model_name = 'AgModel Obesity_risk_v01'

scored_name = 'test_scored_v01.csv'

scored_folder = Path(r'C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\08 ObesityRisk')


scored_path = str(scored_folder / scored_name)


data_folder = Path(r'C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Classification\08 ObesityRisk')
data_name = r'08 ObesityRisk_test.csv'
y_name = "NObeyesdad"

output_col = ['id',y_name]

model_path = model_folder / model_name
data_path = data_folder / data_name

predictor = TabularPredictor.load(model_path)

if ".csv" in data_name:
    data = pd.read_csv(data_path)
elif ".parquet" in data_name:
    data = pd.read_parquet(data_path)



predictions = predictor.predict(data)
data_scored = data.copy()
data_scored[y_name] = predictions

output = data_scored[output_col]

output.to_csv(scored_path,index = False)

