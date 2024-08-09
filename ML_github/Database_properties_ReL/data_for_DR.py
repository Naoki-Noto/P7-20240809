# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd


df_e1_01 = pd.read_csv('data/data_AI.csv')
df_e1_01['made_by'] = 'e=1-0.1'
print(df_e1_01)
 
df_e01 = pd.read_csv('data/data_AI2.csv')
df_e01['made_by'] = 'e=0.1'
print(df_e01)

df_e1 = pd.read_csv('data/data_Random.csv')
df_e1['made_by'] = 'e=1'
print(df_e1)

df_human = pd.read_csv('data/data_Human.csv')
df_human['made_by'] = 'human'
print(df_human)

df_real = pd.read_csv('data/data_real.csv')
df_real['made_by'] = 'real'
print(df_real)

df = pd.concat([df_e1_01, df_e01, df_e1, df_human, df_real])
print(df)

df.to_csv('data/smiles_list_25000.csv', index=False)
