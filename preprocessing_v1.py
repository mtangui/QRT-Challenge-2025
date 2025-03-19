from Tangui_MICHAL_Projet.preprocessing.merger_transformer import *

# from merger_transformer import DataFrameConcatenator, FillNaTransformer, ColumnOrderTransformer

import pandas as pd
from sklearn.pipeline import Pipeline
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# Data Importation:
## Clinical Data
df_clinical = pd.read_csv("./Tangui_MICHAL_Projet/data/X_train/clinical_train.csv")
df_clinical_eval = pd.read_csv("./Tangui_MICHAL_Projet/data/X_test/clinical_test.csv")

## Molecular Data
maf_df = pd.read_csv("./Tangui_MICHAL_Projet/data/X_train/molecular_train.csv")
maf_eval = pd.read_csv("./Tangui_MICHAL_Projet/data/X_test/molecular_test.csv")

## Target Data
target_df = pd.read_csv("./Tangui_MICHAL_Projet/data/target_train.csv")
target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)

# Super Pipeline
super_pipeline = Pipeline([
    ("feature_union", DataFrameConcatenator()),
    ("fill_na", FillNaTransformer()),  # Gestion des NaN
    ("ordering_column", ColumnOrderTransformer())
])

# Transformation des données avec la Super Pipeline
df_transformed = super_pipeline.fit_transform([maf_df, df_clinical]).copy()
df_transformed['ID'] = df_clinical['ID']
df_transformed_eval = super_pipeline.transform([maf_eval, df_clinical_eval]).copy()
df_transformed_eval['ID'] = df_clinical_eval['ID']

# Sauvegarde des données preprocessed
df_transformed.to_pickle("./Tangui_MICHAL_Projet/data/processed/train_preprocess.pkl")
df_transformed_eval.to_pickle("./Tangui_MICHAL_Projet/data/processed/test_preprocess.pkl")

# Sauvegarde de la pipeline
joblib.dump(super_pipeline, "./Tangui_MICHAL_Projet/preprocessing/preprocessing_pipeline.pkl")
