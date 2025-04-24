# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd


df_e01 = pd.read_csv('result/smiles.csv')
df_e01['made_by'] = '1/avgTC'
print(df_e01)
 
df_e01_avgTC = pd.read_csv('result/smiles_avgTC.csv')
df_e01_avgTC['made_by'] = 'avgTC'
print(df_e01_avgTC)

df = pd.concat([df_e01, df_e01_avgTC])
print(df)
df.to_csv('data_to_MF/smiles_list_all.csv', index=False)
