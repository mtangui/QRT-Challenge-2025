from Tangui_MICHAL_Projet.preprocessing.maf_df_transformer import MafDataTransformer
from Tangui_MICHAL_Projet.preprocessing.df_clinical_transformer import CytogeneticsTransformer

# from preprocessing.maf_df_transformer import MafDataTransformer
# from preprocessing.df_clinical_transformer import CytogeneticsTransformer


# from maf_df_transformer import MafDataTransformer
# from df_clinical_transformer import CytogeneticsTransformer

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Concatenation & Transformation finale
## Concatenation Transformer
class DataFrameConcatenator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.maf_pipeline = Pipeline([("maf_transform", MafDataTransformer())])
        self.clinical_pipeline = Pipeline([("cyto_transform", CytogeneticsTransformer())])

    def fit(self, X_list, y=None):
        self.maf_pipeline = self.maf_pipeline.fit(X_list[0])
        self.clinical_pipeline = self.clinical_pipeline.fit(X_list[1])
        return self

    def transform(self, X_list):
        maf_df_transformed = self.maf_pipeline.transform(X_list[0])
        df_clinical_transformed = self.clinical_pipeline.transform(X_list[1])
        return df_clinical_transformed.merge(maf_df_transformed, on='ID', how='left').drop(columns='ID', axis=1)

## Custom Transformer pour gérer les NaN
class FillNaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Remplir NaN : 0 pour les numériques, False pour les booléens
        X[X.select_dtypes(include=['number']).columns] = X.select_dtypes(include=['number']).fillna(0)
        X[X.select_dtypes(include=['bool']).columns] = X.select_dtypes(include=['bool']).fillna(False)
        return X

class ColumnOrderTransformer(BaseEstimator, TransformerMixin):
    """ Assure que les colonnes de test soient ordonnées comme celles du fit """
    def __init__(self):
        self.columns_ = list()

    def fit(self, X, y=None):
        self.columns_ = X.columns  # Stocke l'ordre des colonnes
        return self

    def transform(self, X):
        return X[self.columns_]  # Réordonne X selon l'ordre stocké
