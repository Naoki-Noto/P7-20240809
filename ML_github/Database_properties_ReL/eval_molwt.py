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


data_e1_01 = pd.read_csv('data/data_AI.csv')
MWs_e1_01 = []
for i in range(len(data_e1_01)):
    MWs_e1_01.append([get_molwt(data_e1_01.iloc[i]["SMILES"])])
df_e1_01 = pd.DataFrame(MWs_e1_01, columns=["MolWt"])


data_e1 = pd.read_csv('data/data_Random.csv')
MWs_e1 = []
for i in range(len(data_e1)):
    MWs_e1.append([get_molwt(data_e1.iloc[i]["SMILES"])])
df_e1 = pd.DataFrame(MWs_e1, columns=["MolWt"])


data_e01 = pd.read_csv('data/data_AI2.csv')
MWs_e01 = []
for i in range(len(data_e01)):
    MWs_e01.append([get_molwt(data_e01.iloc[i]["SMILES"])])
df_e01 = pd.DataFrame(MWs_e01, columns=["MolWt"])


molwt_human = df_human["MolWt"]
molwt_human.describe()
molwt_e1_01 = df_e1_01["MolWt"]
molwt_e1_01.describe()
molwt_e1 = df_e1["MolWt"]
molwt_e1.describe()
molwt_e01 = df_e01["MolWt"]
molwt_e01.describe()

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1400)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_e01, bins=100, color="blue", alpha=0.4, label='e01')
ax.hist(molwt_e1, bins=100, color="green", alpha=0.4, label='e1')
ax.hist(molwt_e1_01, bins=100, color="red", alpha=0.4, label='e1_01')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
fig.savefig("result/Eval_MolWt_ReL.pdf")


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1400)
ax.set_ylim(0, 1000)
ax.set_xlabel("MolWt", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.hist(molwt_human, bins=110, color="black", alpha=0.4, label='human')
ax.hist(molwt_e1, bins=100, color="green", alpha=0.4, label='e1_01')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
fig.savefig("result/Eval_MolWt_Random_vs_human.pdf")
