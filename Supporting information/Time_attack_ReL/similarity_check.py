#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:04:41 2024

@author: nata
"""

import pandas as pd

data1 = pd.read_csv('result/smiles_e1.csv')
data2 = pd.read_csv('result/result_ta.csv')

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

smiles_set1 = set(df1['SMILES'])
smiles_set2 = set(df2['SMILES'])

common_smiles = len(smiles_set1.intersection(smiles_set2))
print(f"No. of common SMILES: {common_smiles}")

unique_smiles_in_df1 = smiles_set1.difference(smiles_set2)
print(f"SMILES in only df1: {unique_smiles_in_df1}")

unique_smiles_in_df2 = smiles_set2.difference(smiles_set1)
print(f"SMILES in only df2: {unique_smiles_in_df2}")