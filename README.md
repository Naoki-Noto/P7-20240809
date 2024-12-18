# P7-20240809
===============================================================================

Note: Databases A, B, C, and D are denoted as 'Human', 'Random', 'AI2', and 'AI', respectively.

Note2: Since the pkl files were too large to upload to GitHub, they are stored at the following URL:
https://drive.google.com/drive/folders/1pUC8uVw9CC82m1UwBKBRTU9kUn5yyIWq?usp=sharing

Note3: The environment used is indicated at the end of each directory name.

===============================================================================

# Table of Contents
Database construction/

• Database_properties_ReL: Code and results for comparing chemical spaces and molecular weight distributions.

• Make_database_adapt1: XYZ files after geometry optimization.

HGB: Code for SHAP-based analysis based on HGB models.

data: Code for generating pre-training labels.

• MolGeneration_ReL: Code for reinforcement learning-based molecular generators.

• pkl_files_Deep2: Code for generating pkl files.

==============================================================================

Environment/

For performing machine learning (Deep2.yml): Python (3.10.13) was used as a language, and used packages were deepchem (2.8.0), numpy (1.26.3), pandas (2.2.2), scikit-learn (1.5.0), torch (2.2.0+cu121), torch_geometric (2.4.0), and tensorflow (2.15.0).

For generating pre-training targets (adapt1.yml): Python (3.7.16) was used as a language, and used packages were numpy (1.21.6), matplotlib (3.5.3), mordred (1.2.0), pandas (1.3.5), rdkit (2023.3.2), seaborn (0.12.2), shap (0.42.1), and scikit-learn (1.0.2).

For constructing databases (ReL.yml): Python (3.10.14) was used as a language, and used packages were matplotlib (1.2.0), numpy (1.26.4), pandas (2.2.2), rdkit (2023.9.6), seaborn (0.13.2), scikit-learn (1.4.2), and umap-learn (0.5.6).


==============================================================================

Machine learning/

• Conventional_ML

Lasso: Code and results for Lasso regression.

RF: Code and results for random forest.

RF_control: Code and results for control experiments (random forest).

SVM: Code and results for support-vector machine.

XGB: Code and results for XGBoost.

• DA1: Code and main results of domain adaptation.

• DA2: Code and results of domain adaptation when using 10 data points as a training set.

• DA3_(C4_12): Code and results of domain adaptation for alkene photoisomerization.

Alkene_isomerization_8OPSs: Code and results for predictions in alkene isomerization by domain adaptation using 8 OPSs as the training set.

Alkene_isomerization_DA: Code and results for predictions in alkene isomerization by domain adaptation.

Alkene_isomerization_RF: Code and results for predictions in alkene isomerization by random forest.

Alkene_isomerization_badcase: Code and results for predictions in alkene isomerization by domain adaptation using the source domain with photocatalytic activity trends less similar to the target domain.

Correlation_analysis: Code and results for the correlation analysis.

Learning_curve: Code and results for generating learning curves to investigate generalization performance.

• DA4_(C9_10_15): Code and results of investigations into the limitation and applicability of domain adaptation.

CN: Code and results for domain adaptation in CN.

CO_a: Code and results for domain adaptation in CO_a.

CO_b: Code and results for domain adaptation in CO_b.

CO_c: Code and results for domain adaptation in CO_c.

CO_d: Code and results for domain adaptation in CO_d.

CO_e: Code and results for domain adaptation in CO_e.

CS: Code and results for domain adaptation in CS.

SD_data_exclusion: Code and results for investigations into the influence of excluding 30 OPSs from the source domain.

• DA_SI_(C3_5_13): Code and results of domain adaptation for supporting information.

Comparison_method: Code and results for comparison of DA methods.

Increasing_training_data: Code and results for the test with larger training datasets.

Top3_and_bottom3: Code and results for domain adaptation using source domains selected based on correlation coefficients among OPSs in the training data.

• Data_volume_(C8): Code and results of investigations into the effect of increasing the data volume (but not domain adaptation).

Lasso: Code and results for Lasso regression.

RF: Code and results for random forest.

SVM: Code and results for support-vector machine.

XGB: Code and results for XGBoost.

• Make_descriptors: Code for generating descriptors.

• Paired_t-test: Code and results of paired t-test.


==============================================================================

Supporting information/


