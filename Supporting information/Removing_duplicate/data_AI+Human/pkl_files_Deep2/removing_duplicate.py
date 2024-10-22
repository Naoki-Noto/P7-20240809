# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd
import numpy as np
from rdkit import Chem


data = pd.read_csv('data_AI+Human/data_AI+Human_source.csv')
print(data)
duplicated_smiles = data[data.duplicated(subset='SMILES')]
print(duplicated_smiles)
duplicated_smiles.to_csv('data_AI+Human/duplicated_SMILES_AI+Human.csv', index=False)

data_unique = data.drop_duplicates(subset='SMILES')
print(data_unique)
data_unique.to_csv('data_AI+Human/data_AI+Human.csv', index=False)

smiles_list = np.array(data['SMILES'])
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

unique_mols = {}
for mol in mols:
    canonical_smiles = Chem.MolToSmiles(mol)
    unique_mols[canonical_smiles] = mol

final_smiles_list = list(unique_mols.keys())

f_smiles = pd.DataFrame(final_smiles_list, columns=['SMILES'])
print(f_smiles)
