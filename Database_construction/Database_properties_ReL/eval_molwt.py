from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import matplotlib.pyplot as plt

import pandas as pd

def get_molwt(given):
    molecule = Chem.MolFromSmiles(given)
    molecule = Chem.AddHs(molecule)
    return MolWt(molecule)


data_human = pd.read_csv('data/data_Human.csv')
MWs_human = []
for i in range(len(data_human)):
    MWs_human.append([get_molwt(data_human.iloc[i]["SMILES"])])
df_human = pd.DataFrame(MWs_human, columns=["MolWt"])
print(df_human["MolWt"].max())
print(df_human["MolWt"].min())


data_e1_01 = pd.read_csv('data/data_AI.csv')
MWs_e1_01 = []
for i in range(len(data_e1_01)):
    MWs_e1_01.append([get_molwt(data_e1_01.iloc[i]["SMILES"])])
df_e1_01 = pd.DataFrame(MWs_e1_01, columns=["MolWt"])
print(df_e1_01["MolWt"].max())
print(df_e1_01["MolWt"].min())

data_e1 = pd.read_csv('data/data_Random.csv')
MWs_e1 = []
for i in range(len(data_e1)):
    MWs_e1.append([get_molwt(data_e1.iloc[i]["SMILES"])])
df_e1 = pd.DataFrame(MWs_e1, columns=["MolWt"])
print(df_e1["MolWt"].max())
print(df_e1["MolWt"].min())


data_e01 = pd.read_csv('data/data_AI2.csv')
MWs_e01 = []
for i in range(len(data_e01)):
    MWs_e01.append([get_molwt(data_e01.iloc[i]["SMILES"])])
df_e01 = pd.DataFrame(MWs_e01, columns=["MolWt"])
print(df_e01["MolWt"].max())
print(df_e01["MolWt"].min())

molwt_human = df_human["MolWt"]
molwt_human.describe()
molwt_e1_01 = df_e1_01["MolWt"]
molwt_e1_01.describe()
molwt_e1 = df_e1["MolWt"]
molwt_e1.describe()
molwt_e01 = df_e01["MolWt"]
molwt_e01.describe()

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_e1, bins=88, color="green", alpha=0.4, label='e1')
ax.hist(molwt_e1_01, bins=87, color="red", alpha=0.4, label='e1_01')
plt.show()
fig.savefig("result/Eval_MolWt_ReL_1.pdf")

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_e01, bins=86, color="blue", alpha=0.4, label='e01')
ax.hist(molwt_e1, bins=88, color="green", alpha=0.4, label='e1')
plt.show()
fig.savefig("result/Eval_MolWt_ReL_2.pdf")

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_e01, bins=86, color="blue", alpha=0.4, label='e01')
ax.hist(molwt_e1, bins=88, color="green", alpha=0.4, label='e1')
ax.hist(molwt_e1_01, bins=87, color="red", alpha=0.4, label='e1_01')
plt.show()
fig.savefig("result/Eval_MolWt_ReL_3.pdf")

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_human, bins=113, color="black", alpha=0.4, label='human')
ax.hist(molwt_e1, bins=88, color="green", alpha=0.4, label='e1')
plt.show()
fig.savefig("result/Eval_MolWt_Random_vs_human.pdf")

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_human, bins=113, color="black", alpha=0.4, label='human')
ax.hist(molwt_e1_01, bins=87, color="red", alpha=0.4, label='e1_01')
plt.show()
fig.savefig("result/Eval_MolWt_AI_vs_human.pdf")
