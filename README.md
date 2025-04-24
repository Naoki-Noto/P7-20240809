# P7-20240809
===========================================================================

Note: Databases A, B, C, D, and E are denoted as 'Human', 'Random', 'AI2', 'AI', and 'AI2+Human', respectively.

Note2: Since the pkl files were too large to upload to GitHub, they are stored at the following URL (data_AI: Database D, data_AI2+Human: Database E, data_AI2: Database C, data_Human: Database A, data_Random: Database B, data_zinc_25286/data_zinc_50572: ZINC-based databases):
https://drive.google.com/drive/folders/1pUC8uVw9CC82m1UwBKBRTU9kUn5yyIWq?usp=sharing

Note3: The environment used is indicated at the end of each directory name.

===========================================================================

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

- MolGeneration_SI_ReL/MolGeneration_SI2_ReL: Code for comparing molecular properties derived from policy and reward settings.

- Removing_duplicate: Code and results for investigating the effect of removing duplicates in Databases B and E on predictive performance.

- Time_attack_ReL: Code for measuring the time required to construct Database B.

- ZINC_database: Code and results when using ZINC-derived databases.
