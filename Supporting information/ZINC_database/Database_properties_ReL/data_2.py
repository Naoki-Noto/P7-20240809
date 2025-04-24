# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd


df_e = pd.read_csv('data_to_MF/data_AI2+Human.csv')
df_e['made_by'] = 'E'
print(df_e)
 
df_zinc = pd.read_csv('data_to_MF/data_zinc_50572.csv')
df_zinc['made_by'] = 'zinc'
print(df_zinc)

df = pd.concat([df_e, df_zinc])
print(df)

df.to_csv('data_to_MF/smiles_list_2.csv', index=False)
