import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MafDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nbr_max_duplicate=10):
        self.nbr_max_duplicate = nbr_max_duplicate
        self.list_gene = None
        self.list_effect = None
        self.chr_list = ["X", "Y"] + [str(i) for i in range(1, 23)]

    def fit(self, X, y=None):
        self.list_gene = list(X["GENE"].unique()) + ['UNKNOWN']
        self.list_effect = list(X["EFFECT"].unique()) + ['UNKNOWN']
        return self

    def transform(self, X):
        maf_df = X.copy()
        maf_df = maf_df.dropna(subset=['CHR'])  # Supprime les lignes où 'CHR' est NaN
        maf_df = maf_df[maf_df['CHR'].astype(str).str.strip() != ""]  # Supprime les lignes où 'CHR' est une chaîne vide

        maf_df['UNIQUE_KEY'] = maf_df['START'] + maf_df['END'] + maf_df['VAF'] + maf_df['DEPTH']
        maf_df['UNIQUE_KEY'] = maf_df['UNIQUE_KEY'].apply(hash)
        maf_df.drop(index=maf_df[maf_df['UNIQUE_KEY'] == 0].index, axis=0, inplace=True)
        maf_df.drop(columns=['UNIQUE_KEY'], axis=1, inplace=True)

        values = {'EFFECT': 'UNKNOWN',
                  }
        maf_df.fillna(value=values, inplace=True)

        tmp = pd.DataFrame({"ID": maf_df["ID"].unique()})
        tmp.set_index("ID", inplace=True)

        new_columns = {}
        n = len(tmp.index)

        for effect in self.list_effect:
            new_columns[f'EFFECT_{effect}'] = [0] * n

        for gene in self.list_gene:
            new_columns[f'GENE_{gene}'] = [0] * n

        column_name = ["START", "END", "END-START", "VAF", "DEPTH"]

        for chromosome in self.chr_list:
            for name in column_name:
                for i in range(self.nbr_max_duplicate):
                    new_columns[f"CHR_{chromosome}_{name}_{i}"] = [0] * n

        tmp = pd.concat([tmp, pd.DataFrame(new_columns, index=tmp.index)], axis=1)
        tmp = tmp.copy()

        tmp['Nmut'] = maf_df.groupby('ID').size().reset_index(name='Nmut').fillna({'Nmut': 0}).set_index('ID')

        # Créer un dictionnaire pour stocker les indices de duplication par ID_Patient et CHR
        duplication_indices = {}

        # Parcourir le DataFrame maf_df efficacement avec iterrows()
        for index, row in maf_df.iterrows():
            ID, CHR, START, END, REF, ALT, GENE, PROTEIN_CHANGE, EFFECT, VAF, DEPTH = row

            # Incrémente les colonnes "GENE" et "EFFECT"
            if f'GENE_{GENE}' not in tmp.columns:
                if f'GENE_{GENE}' in self.list_gene:
                    tmp[f'GENE_{GENE}'] = 0
                else:
                    tmp.loc[ID, 'GENE_UNKNOWN'] += 1

            if f'GENE_{GENE}' in self.list_gene:
                tmp.loc[ID, f'GENE_{GENE}'] += 1
            else:
                tmp.loc[ID, 'GENE_UNKNOWN'] += 1

            if f'EFFECT_{EFFECT}' not in tmp.columns:
                tmp[f'EFFECT_{EFFECT}'] = 0
            tmp.loc[ID, f'EFFECT_{EFFECT}'] += 1

            # Obtenir l'index de duplication sans utiliser de boucle while
            key = (ID, CHR)
            if key not in duplication_indices:
                duplication_indices[key] = 0
            else:
                duplication_indices[key] += 1

            nbr_dupli = duplication_indices[key]

            # Assigner directement les valeurs aux colonnes
            tmp.loc[ID, [f'CHR_{CHR}_START_{nbr_dupli}', f'CHR_{CHR}_END_{nbr_dupli}', f'CHR_{CHR}_END-START_{nbr_dupli}', f'CHR_{CHR}_VAF_{nbr_dupli}', f'CHR_{CHR}_DEPTH_{nbr_dupli}']] = START, END, END-START, VAF, DEPTH

        return tmp



