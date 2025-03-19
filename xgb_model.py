import xgboost as xgb
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

xgb_model = xgb.XGBRegressor(max_depth=1)
xgb_model.fit(X, y)

# Sauvegarde du modèle
joblib.dump(xgb_model, "./Tangui_MICHAL_Projet/models/xgb_model.pkl")
