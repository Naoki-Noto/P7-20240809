# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd
import numpy as np
from rdkit import Chem

epsilon, R = 1, 10
#epsilon, R = '1_01', 4
#epsilon, R = '01', 9

def number_bond_silicon(mol):
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'Si':
                if atom.GetDegree() > 3:
                    return False
                if atom.GetDegree() == 2 and atom.GetNumRadicalElectrons() > 0:
                    return False
    return True

incorrect_patterns = ["[SiH2]([*])([*])([*])"]

def remove_incorrect_silicon(molecules, patterns):
    filtered_molecules = []
    for mol in molecules:
        if mol is None:
            continue
        remove = False
        for pattern in patterns:
            patt = Chem.MolFromSmarts(pattern)
            if mol.HasSubstructMatch(patt):
                remove = True
                break
        if not remove:
            filtered_molecules.append(mol)
    return filtered_molecules

def number_bond_phosphorus(mol):
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'P':
                if atom.GetDegree() > 4 and atom.GetNumExplicitHs() > 0:
                    return False
    return True


smiles_frames = []
reward_frames = []
for seed in range(R):
    data = pd.read_csv(f'generated_smiles/e{epsilon}/smiles_seed{seed}.csv')
    smiles_frame = pd.DataFrame(data['SMILES'], columns=['SMILES'])
    reward_frame = pd.DataFrame(data['reward'], columns=['reward'])
    
    smiles_frames.append(smiles_frame)
    reward_frames.append(reward_frame)

smiles_df = pd.concat(smiles_frames).reset_index(drop=True)
reward_df = pd.concat(reward_frames).reset_index(drop=True)

smiles_reward = pd.concat([smiles_df, reward_df], axis=1, join='inner')
print(smiles_reward)

df = smiles_reward[smiles_reward['reward'] > 0]
print(df)

smiles_list = np.array(df['SMILES'])
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

mols = [mol for mol in mols if number_bond_silicon(mol)]
mols = [mol for mol in mols if number_bond_phosphorus(mol)]
mols = remove_incorrect_silicon(mols, incorrect_patterns)

unique_mols = {}
for mol in mols:
    canonical_smiles = Chem.MolToSmiles(mol)
    unique_mols[canonical_smiles] = mol

final_smiles_list = list(unique_mols.keys())

f_smiles = pd.DataFrame(final_smiles_list, columns=['SMILES'])
print(f_smiles)
f_smiles.to_csv(f'result/smiles_e{epsilon}.csv', index=False)
