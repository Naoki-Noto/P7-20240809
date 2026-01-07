# P7-20240809

Note: Databases A, B, C, D, E, and F are denoted as 'Human', 'Random', 'AI2', 'AI', 'AI2+Human', and 'AI+Random' respectively.

Note2: Since the pkl files were too large to upload to GitHub, they are stored at the following URL (data_AI: Database D, data_AI2+Human: Database E, data_AI2: Database C, data_Human: Database A, data_Random: Database B, data_zinc_25286/data_zinc_50572: ZINC-based databases):
https://drive.google.com/drive/folders/1pUC8uVw9CC82m1UwBKBRTU9kUn5yyIWq?usp=sharing

Note3: The environment used is indicated at the end of each directory name.

# Citation Information

title={Transfer learning from custom-tailored virtual molecular databases to real-world organic photosensitizers for catalytic activity prediction}

author={Naoki Noto, Taiki Nagano, Mikito Fujinami, Ryosuke Kojima, Susumu Saito }

journal info={Commun. Chem. 2025, 8, Article number: 288.}

DOI: https://doi.org/10.1038/s42004-025-01678-w

# Table of Contents
Database construction/

- Database_properties_ReL: Code and results for comparing chemical spaces and molecular weight distributions.

- Make_database_adapt1: Code and results for making databases.

  • HGB: Code for SHAP-based analysis based on HGB models.

  • data: Code for generating pre-training labels.

- MolGeneration_ReL: Code for reinforcement learning-based molecular generators.

- Pubchem_ReL: Code and results for checking whether molecules are registered in PubChem.

- pkl_files_Deep2: Code for generating pkl files.

==========================================================================

Environment: Environments for performing each code are stored in this directory./

For performing machine learning (Deep2.yml): Python (3.10.13) was used as a language, and used packages were deepchem (2.8.0), matplotlib (3.9.0), numpy (1.26.3), pandas (2.2.2), scikit-learn (1.5.0), tensorflow (2.15.0), torch (2.2.0+cu121), and torch_geometric (2.4.0).

For using ReactionT5 (reactiont5.yml): Python (3.10.16) was used as a language, and used packages were numpy (2.2.1), pandas (2.2.3), rdkit (2024.3.6), scikit-learn (1.6.0), torch (2.5.1+cu121), and transformers (4.47.1).

For generating pre-training targets (adapt1.yml): Python (3.7.16) was used as a language, and used packages were matplotlib (3.5.3), mordred (1.2.0), numpy (1.21.6), pandas (1.3.5), rdkit (2023.3.2), scikit-learn (1.0.2), seaborn (0.12.2), and shap (0.42.1).

For constructing databases (ReL.yml): Python (3.10.14) was used as a language, and used packages were matplotlib (3.9.0), numpy (1.26.4), pandas (2.2.2), pubchempy(1.0.4), rdkit (2023.9.6), seaborn (0.13.2), scikit-learn (1.4.2), and umap-learn (0.5.6).


==========================================================================

Machine learning/

- Benchmark_Deep2: Code and results for benchmark models, including Random forest (RDKit descriptor/Mordred) and GCN models.

- DL_Deep2: Code and results of deep learning studies.

  • BertzCT/BertzCT_add: Code and results for constructing pre-trained models based on BertzCT and fine-tuning.

  • TargetScreening_yield_s/TargetScreening_yield_l_cl: Code and results for constructing pre-trained models based on various pre-training labels and fine-tuning. Database B was utilized for TargetScreening_yield_s as well as Database E for TargetScreening_yield_l_cl.

- reactiont5_reactiont5: Code and results of ReactionT5.

==========================================================================

Supporting information/

- Chemical_space_B&E_ReL: Code and results for comparing chemical spaces of Databases B and E.
  
- MolGeneration_SI_ReL: Code for comparing molecular properties derived from policy and reward settings.

- Molecule_selection_ReL: Code and results for randomly selecting 10 molecules from Databases A, B, C, and D.

- Removing_duplicate: Code and results for investigating the effect of removing duplicates in Databases B and E on predictive performance.

- Time_attack_ReL: Code for measuring the time required to construct Database B.

- ZINC_database: Code and results when using ZINC-derived databases.

# Setup and Usage
**1. For using molecular generator**

To set up the environment for using the molecular generator, please execute the following command using the `ReL.yml` file located in the Environment folder. If some packages are not installed successfully, please refer to the `ReL.yml` file and install the missing packages manually using `pip`.
```bash
conda env create -n new_env -f ReL.yml
conda activate new_env
```
Next, run one of the following scripts based on your intended setup:  
- `QL_e01.py` # Molecular generator based on epsiron = 0.1
- `QL_e1.py` # Molecular generator based on epsiron = 1
- `QL_e1_01.py` # Molecular generator with epsilon annealing (from 1 to 0.1)

These scripts are located in:
```bash
Database construction/
└── MolGeneration_ReL/
  ├── QL_e01.py
  ├── QL_e1.py
  ├── QL_e1_01.py
  ├── QL_env_agent_e01.py # Required file to run QL_e01.py
  ├── QL_env_agent_e1.py # Required file to run QL_e1.py
  └── QL_env_agent_e1_01.py # Required file to run QL_e1_01.py
```

**2. For preparing pretraining labels**

To set up the environment for preparing pretraining labels, please execute the following command using the `adapt1.yml` file located in the Environment folder. If some packages are not installed successfully, please refer to the `adapt1.yml` file and install the missing packages manually using `pip`.
```bash
conda env create -n new_env2 -f adapt1.yml
conda activate new_env2
```
Next, run one of the following scripts based on your intended setup:  
- `RDKit_AI.ipynb`,`RDKit_AI2.ipynb`,`RDKit_human.ipynb`,`RDKit_random.ipynb` # To generate RDKit descriptor
- `mordred_AI.ipynb`,`mordred_AI2.ipynb`,`mordred_human.ipynb`,`mordred_random.ipynb` # To generate Mordred descriptor
- `summary_AI.ipynb`,`summary_AI2.ipynb`,`summary_human.ipynb`,`summary_random.ipynb` # To extract necessary labels, check for NaN values, and perform random sampling

These scripts are located in:
```bash
Database construction/
└── Make_database_adapt1/
  └── data/
    ├── RDKit_AI.ipynb
    ├── RDKit_AI2.ipynb
    ├── RDKit_human.ipynb
    ├── RDKit_random.ipynb
    ├── mordred_AI.ipynb
    ├── mordred_AI2.ipynb
    ├── mordred_human.ipynb
    ├── summary_AI.ipynb
    ├── summary_AI2.ipynb
    ├── summary_human.ipynb
    ├── summary_random.ipynb
    ├── result/ # This folder is used for storing the output results.
    └── source/ # This folder includes SMILES lists to use the generation of pretraining labels.
```

**3. For conducting transfer learning**

To set up the environment for conducting transfer learning, please execute the following command using the `Deep2.yml` file located in the Environment folder. If some packages are not installed successfully, please refer to the `Deep2.yml` file and install the missing packages manually using `pip`.
```bash
conda env create -n new_env3 -f Deep2.yml
conda activate new_env3
```
Next, to make pkl files, run one of the following scripts based on your intended setup: 
- `Make_dataset_AI+AI2.ipynb`,`Make_dataset_AI+Human.ipynb`,`Make_dataset_AI+Random.ipynb`,`Make_dataset_AI.ipynb`,`Make_dataset_AI2+Human.ipynb`,`Make_dataset_AI2+Random.ipynb`,`Make_dataset_AI2.ipynb`,`Make_dataset_Human.ipynb`,`Make_dataset_Random+Human.ipynb`,`Make_dataset_Random.ipynb`, # To make pkl files

These scripts are located in:
```bash
Database construction/
└── pkl_files_Deep2/
  ├── Make_dataset_AI+AI2.ipynb
  ├── Make_dataset_AI+Human.ipynb
  ├── Make_dataset_AI+Random.ipynb
  ├── Make_dataset_AI.ipynb
  ├── Make_dataset_AI2+Human.ipynb
  ├── Make_dataset_AI2+Random.ipynb
  ├── Make_dataset_AI2.ipynb
  ├── Make_dataset_Human.ipynb
  ├── Make_dataset_Random+Human.ipynb
  ├── Make_dataset_Random.ipynb
  ├── data_AI+AI2/
  ├── data_AI+Human/
  ├── data_AI+Random/
  ├── data_AI/
  ├── data_AI2+Human/
  ├── data_AI2+Random/
  ├── data_AI2/
  ├── data_Human/
  ├── data_Random+Human/
  └── data_Random/
```
Next, to perform deep learning, run one of the following scripts based on your intended setup:  
- `GCN.ipynb` # To perform supervised pretraining
- `FT_yield_s.ipynb`,`FT_yield_l.ipynb`,`FT_yield_cl.ipynb`,`FT_yield_CS.ipynb`,`FT_yield_CN.ipynb`,`FT_yield_CA.ipynb` # To perform fine-tuning for CO-a, CO-b, CO-c, CS, CN, and CA, respectively
  
These scripts are located in:
```bash
Machine learning/
└── DeepLearning_Deep2/
  ├── ABCGG_add/
  │ ├── FT_yield_CA.ipynb
  │ ├── FT_yield_CS.ipynb
  │ ├── GCN.ipynb
  │ ├── data_AI+AI2/
  │ ├── data_AI+Human/
  │ ├── data_AI+Random/
  │ ├── data_AI2+Human/ # This dataset is the best for pretraining.
  │ ├── data_AI2+Random/
  │ ├── data_Random+Human/
  │ ├── data_real/ # This folder is used for storing the output results.
  │ └── result/
  ├── BertzCT/
  │ ├── FT_yield_s.ipynb
  │ ├── GCN.ipynb
  │ ├── data_AI/
  │ ├── data_AI2/
  │ ├── data_human/
  │ ├── data_random/ # This dataset is the best for pretraining.
  │ ├── data_real/
  │ └── result/ # This folder is used for storing the output results.
  ├── BertzCT_add/
  │ ├── FT_yield_l.ipynb
  │ ├── FT_yield_cl.ipynb
  │ ├── GCN.ipynb
  │ ├── data_AI+AI2/
  │ ├── data_AI+Human/
  │ ├── data_AI+Random/
  │ ├── data_AI2+Human/ # This dataset is the best for pretraining.
  │ ├── data_AI2+Random/
  │ ├── data_Random+Human/
  │ ├── data_real/
  │ └── result/ # This folder is used for storing the output results.
  └── Kappa3_add/
    ├── FT_yield_CN.ipynb
    ├── GCN.ipynb
    ├── data_AI+AI2/
    ├── data_AI+Human/
    ├── data_AI+Random/ # This dataset is the best for pretraining.
    ├── data_AI2+Human/
    ├── data_AI2+Random/
    ├── data_Random+Human/
    ├── data_real/
    └── result/ # This folder is used for storing the output results.
```
