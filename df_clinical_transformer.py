import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class CytogeneticsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.expected_columns_center = None

    def extract_cytogenetics(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return {"chromosome_count": None, "sex": None}

        chromosome_count_match = re.search(r"(\d{2})", text)
        chromosome_count = int(chromosome_count_match.group(1)) if chromosome_count_match else None

        sex_match = re.search(r"(xx|xy)", text)
        sex = sex_match.group(1) if sex_match else None

        return {"chromosome_count": chromosome_count, "sex": sex}

    def fit(self, X, y=None):
        self.expected_columns_center = list(pd.get_dummies(X, columns=['CENTER']).columns) + ['CENTER_UNKNOWN']
        return self

    def transform(self, X):
        df = X.copy()

        df = pd.get_dummies(df, columns=['CENTER'])
        for col in self.expected_columns_center:
            if col not in df.columns:
                df[col] = 0.0

        for col in df.columns:
            if col not in self.expected_columns_center:
                df['CENTER_UNKNOWN'] += df[col]
                df.drop(col, axis=1, inplace=True)

        values = {'WBC': df['WBC'].mean(),
                  'ANC': df['ANC'].mean(),
                  'MONOCYTES': df['MONOCYTES'].mean(),
                  'BM_BLAST': df['BM_BLAST'].mean(),
                  'HB': df['HB'].mean(),
                  'PLT': df['PLT'].mean()
                  }
        df.fillna(value=values, inplace=True)

        df_parsed = df['CYTOGENETICS'].apply(self.extract_cytogenetics)
        df = df.join(pd.DataFrame(df_parsed.to_list()))

        df['sex'] = df['sex'].apply(lambda x: int(str(x) == 'xy'))

        anomalies = {
            'NORMAL_CARYOTYPE': ['46,xy[20]', '46,xx[20]'],
            'MONOSOMIE_7': ['-7', 'del(7)', 'del(7q)', 'del7', 'del7q'],
            'MONOSOMIE_21': ['-21', 'del(21)', 'del(21q)', 'del21', 'del21q'],
            'TRISOMIE_8': ['+8', 'trisomy8', 'add(8)', 'plus8', 'tris8'],
            'TRISOMIE_21': ['+21', 'add(21)'],
            'TRISOMIE_13': ['+13', 'add(13)'],
            'TRISOMIE_6': ['+6', 'add(6)'],
            'TRISOMIE_11': ['+11', 'add(11)'],
            'MONOSOMIE_DEL_5Q': ['del(5q)', '-5', 'del(5)', 'del5', 'del5q'],
            'DEL_9P': ['del(9p)', 'del(9)', '-9', 'del9', 'del9p'],
            'DEL_20Q': ['del(20q)', 'del(20)', '-20', 'del20', 'del20q'],
            'T_9_22_BCR_ABL1': ['t(9;22)', 'BCR-ABL1'],
            'T_15_17_PML_RARA': ['t(15;17)', 'PML-RARA'],
            'T_8_21_RUNX1_RUNX1T1': ['t(8;21)', 'RUNX1-RUNX1T1'],
            'INV_16_CBFB_MYH11': ['inv(16)', 't(16;16)', 'CBFB-MYH11'],
            'COMPLEXE_CARYOTYPE': ['>3abnormalities', 'complex', 'complexkaryptype', 'complexkaryotype'],
        }

        for key, patterns in anomalies.items():
            df[key] = df['CYTOGENETICS'].apply(lambda x: int(any(p in str(x) for p in patterns)))

        df['chromosome_count'] = df['chromosome_count'].fillna(46)

        return df.drop(columns=['CYTOGENETICS'])
