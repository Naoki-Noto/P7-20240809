# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd

 
df_E = pd.read_csv('data/data_AI2+Human.csv')
df_E['made_by'] = 'E'
print(df_E)

df_e1 = pd.read_csv('data/data_Random.csv')
df_e1['made_by'] = 'e=1'
print(df_e1)

df = pd.concat([df_E, df_e1])
print(df)

df.to_csv('data_to_MF/smiles_list_all.csv', index=False)
