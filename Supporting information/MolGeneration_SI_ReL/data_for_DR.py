# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd


df_e01 = pd.read_csv("result/SMILES_summary/smiles_e01_2400.csv")
df_e01['made_by'] = 'e=0.1_1/T'
print(df_e01)

df_e01_2 = pd.read_csv("result/SMILES_summary/smiles_e01_2_2400.csv")
df_e01_2['made_by'] = 'e=0.1_T'
print(df_e01_2)

df = pd.concat([df_e01,  df_e01_2])
print(df)

df.to_csv('result/smiles_list_2400.csv', index=False)
