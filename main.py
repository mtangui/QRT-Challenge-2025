from Tangui_MICHAL_Projet.preprocessing.merger_transformer import *

# from preprocessing import DataFrameConcatenator, FillNaTransformer, ColumnOrderTransformer

import pandas as pd
from sklearn.pipeline import Pipeline
import warnings
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import xgboost as xgb
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw

warnings.filterwarnings("ignore", category=FutureWarning)


# Data Importation
## Clinical Data
df_clinical = pd.read_csv("./Tangui_MICHAL_Projet/data/X_train/clinical_train.csv")

## Molecular Data
maf_df = pd.read_csv("./Tangui_MICHAL_Projet/data/X_train/molecular_train.csv")

## Target Data
target_df = pd.read_csv("./Tangui_MICHAL_Projet/data/target_train.csv")
target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)

# Train_Test_Split
df_clinical_train, df_clinical_test, y_train, y_test = train_test_split(df_clinical.loc[df_clinical['ID'].isin(target_df['ID'])], y, test_size=0.3, random_state=42)
maf_df_train, maf_df_test = maf_df.loc[maf_df['ID'].isin(df_clinical_train['ID'])], maf_df.loc[maf_df['ID'].isin(df_clinical_test['ID'])]


# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ("feature_union", DataFrameConcatenator()),
    ("fill_na", FillNaTransformer()),  # Gestion des NaN
    ("ordering_column", ColumnOrderTransformer()),
    ("xgb", xgb.XGBRegressor(max_depth=1))
])

# GradientBoostingSurvivalAnalysis Pipeline
gbsa_pipeline = Pipeline([
    ("feature_union", DataFrameConcatenator()),
    ("fill_na", FillNaTransformer()),  # Gestion des NaN
    ("ordering_column", ColumnOrderTransformer()),
    ("gbsa", GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])


gbsa_pipeline.fit([maf_df_train, df_clinical_train], y_train)
pred_train_gbsa = gbsa_pipeline.predict([maf_df_train, df_clinical_train])
pred_test_gbsa = gbsa_pipeline.predict([maf_df_test, df_clinical_test])

train_ci_ipcw = concordance_index_ipcw(y_train,y_train, pred_train_gbsa, tau=7)[0]
test_ci_ipcw = concordance_index_ipcw(y_train, y_test, pred_test_gbsa, tau=7)[0]

print(f"GBSA Model on train: {train_ci_ipcw:.2f}")
print(f"GBSA Model on test: {test_ci_ipcw:.2f}")


xgb_pipeline.fit([maf_df_train, df_clinical_train], y_train['OS_YEARS'])
pred_train_xgb = -xgb_pipeline.predict([maf_df_train, df_clinical_train])
pred_test_xgb = -xgb_pipeline.predict([maf_df_test, df_clinical_test])

train_ci_ipcw = concordance_index_ipcw(y_train,y_train, pred_train_xgb, tau=7)[0]
test_ci_ipcw = concordance_index_ipcw(y_train, y_test, pred_test_xgb, tau=7)[0]

print(f"XGBoost Model on train: {train_ci_ipcw:.2f}")
print(f"XGBoost Model on test: {test_ci_ipcw:.2f}")
