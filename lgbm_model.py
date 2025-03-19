import lightgbm as lgb
import pandas as pd
from sksurv.util import Surv
import joblib

target_df = pd.read_csv("./Tangui_MICHAL_Projet/data/target_train.csv")
target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)

df_transformed = pd.read_pickle("./Tangui_MICHAL_Projet/data/processed/train_preprocess.pkl")

# Create the survival data format
X = df_transformed.loc[df_transformed['ID'].isin(target_df['ID'])].drop('ID', axis=1)
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)
y = y['OS_YEARS']

lgbm_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'verbose': -1
}

train_dataset = lgb.Dataset(X, label=y)

# Train the LightGBM model
lgbm_model = lgb.train(params=lgbm_params, train_set=train_dataset)

# Sauvegarde du modèle
joblib.dump(lgbm_model, "./Tangui_MICHAL_Projet/models/lgbm_model.pkl")
