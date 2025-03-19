from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import warnings
import io

warnings.filterwarnings("ignore", category=FutureWarning)


app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


# Fonction de chargement des modèles dynamiquement
def load_model(model_type: str):
    model_paths = {
        "gbsa": "Tangui_MICHAL_Projet/models/gbsa_model.pkl",
        "xgb": "Tangui_MICHAL_Projet/models/xgb_model.pkl",
        "lgb": "Tangui_MICHAL_Projet/models/lgb_model.pkl",
    }

    if model_type not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose from 'gbsa', 'xgb', or 'lgb'.")

    try:
        model = joblib.load(model_paths[model_type])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    return model


# Chargement du pipeline de preprocessing
def load_pipeline():
    pipeline_path = "./Tangui_MICHAL_Projet/preprocessing/preprocessing_pipeline.pkl"
    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading preprocessing pipeline: {str(e)}")

    return pipeline


# Chargement unique du pipeline (évite de le recharger à chaque requête)
preprocessing_pipeline = load_pipeline()


# Définition des classes Pydantic pour la validation stricte des données
class MafData(BaseModel):
    ID: List[str]
    CHR: List[str]
    START: List[float]
    END: List[float]
    REF: List[str]
    ALT: List[str]
    GENE: List[str]
    PROTEIN_CHANGE: List[str]
    EFFECT: List[str]
    VAF: List[float]
    DEPTH: List[float]


class ClinicalData(BaseModel):
    ID: List[str]
    CENTER: List[str]
    BM_BLAST: List[float]
    WBC: List[float]
    ANC: List[float]
    MONOCYTES: List[float]
    HB: List[float]
    PLT: List[float]
    CYTOGENETICS: List[str]


# Endpoint de prédiction avec sélection du modèle
@app.post("/predict/")
def predict(
        maf_df_data: MafData,
        df_clinical_data: ClinicalData,
        model_type: Literal["gbsa", "xgb", "lgb"] = Query("gbsa",
                                                          description="Select the model: 'gbsa', 'xgb', or 'lgb'"),
):
    """
    Endpoint for model prediction.

    :param data: Input of type `PredictionInput`, validated by Pydantic.
    :param model_type: Choice of model ('gbsa', 'xgb', or 'lgb').

    :return: Risk score (`prediction`).
    """
    try:
        # maf_df = pd.DataFrame.from_dict({key: [value] for key, value in maf_df_data.model_dump().items()})
        maf_df = pd.DataFrame.from_dict(maf_df_data.model_dump())
        # df_clinical = pd.DataFrame.from_dict({key: [value] for key, value in df_clinical_data.model_dump().items()})
        df_clinical = pd.DataFrame.from_dict(df_clinical_data.model_dump())

        # Chargement dynamique du modèle sélectionné
        model = load_model(model_type)

        # Préprocessing des données avec la pipeline
        df_preprocessed = preprocessing_pipeline.transform([maf_df, df_clinical]).copy()

        # Prédiction
        prediction = model.predict(df_preprocessed).tolist()

        return {
            "model_used": model_type,
            "prediction ('ID', prediction)": [(df_clinical['ID'].iloc[i], prediction[i]) for i in range(len(prediction))]
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation Error: {e.errors()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
