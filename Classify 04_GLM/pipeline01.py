# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:12:08 2023

@author: Heng2020
"""
import pandas as pd

# Read columns A to G from the Excel file
excel_path = r"C:\Users\Heng2020\OneDrive\Python Modeling\Modeling 01\Dataset Regression\08 DatesAndGrouping.xlsx"
df = pd.read_excel(excel_path, usecols='A:B')
