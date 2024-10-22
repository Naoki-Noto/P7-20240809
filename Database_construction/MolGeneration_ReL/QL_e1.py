#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:03:43 2024

@author: noton
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from QL_env_agent_e1 import ChemicalSpace, QLAgent

env = ChemicalSpace()
agent = QLAgent()

episodes = 10000

seed = 3
np.random.seed(seed)

smiles_list = []
rewards = []
for episode in range(episodes):
    print(episode)
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if env.op_count > 1:
            mol = Chem.MolFromSmiles(env.smiles)
            #print(env.smiles,reward)
            if mol is not None:
                smiles_list.append(env.smiles)
                rewards.append(reward)
            
        if done:
            break

        state = next_state

df = pd.DataFrame({'SMILES': smiles_list, 'reward': rewards})
print(df)
df.to_csv(f'generated_smiles/e1/smiles_seed{seed}.csv', index=False)
    