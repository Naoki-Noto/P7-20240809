# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd


df_e1 = pd.read_csv('data_to_MF/data_random.csv')
df_e1['made_by'] = 'Random'
print(df_e1)
 
df_zinc = pd.read_csv('data_to_MF/data_zinc_25286.csv')
df_zinc['made_by'] = 'zinc'
print(df_zinc)

df = pd.concat([df_e1, df_zinc])
print(df)

df.to_csv('data_to_MF/smiles_list_1.csv', index=False)
