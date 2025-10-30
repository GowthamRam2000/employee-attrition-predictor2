import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class HRDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.column_defaults = {}

    def load_data(self, filepath):
        try:

            if hasattr(filepath, 'read'):
                if filepath.name.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.name.endswith('.xlsx') or filepath.name.endswith('.xls'):
                    df = pd.read_excel(filepath)
                else:
                    raise ValueError("Unsupported file format")
            else:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                    df = pd.read_excel(filepath)
                else:
                    raise ValueError("Unsupported file format")

            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def preprocess_data(self, df, is_training=True):


        df = df.copy()


        columns_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)


        df_before_encoding = df.copy()


        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()


        if 'Attrition' in categorical_columns:
            categorical_columns.remove('Attrition')


        for col in categorical_columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:

                    le = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0]
                    if x in le.classes_ else -1)
                else:

                    df = df.drop(columns=[col])


        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
            y = df['Attrition'].values
            X = df.drop('Attrition', axis=1)
        else:
            y = None
            X = df


        if is_training:
            self.feature_columns = X.columns.tolist()


            defaults = {}
            raw_df = df_before_encoding.copy()
            if 'Attrition' in raw_df.columns:
                raw_df = raw_df.drop('Attrition', axis=1)

            for col in raw_df.columns:
                if raw_df[col].dtype == object:

                    try:
                        defaults[col] = raw_df[col].mode(dropna=True).iloc[0]
                    except Exception:
                        defaults[col] = ''
                else:

                    try:
                        defaults[col] = float(raw_df[col].median())
                    except Exception:
                        defaults[col] = 0.0
            self.column_defaults = defaults


        if not is_training and self.feature_columns is not None:

            missing = [c for c in self.feature_columns if c not in X.columns]
            for col in missing:
                if col in self.label_encoders:
                    X[col] = -1
                else:
                    X[col] = 0

            extra = [c for c in X.columns if c not in self.feature_columns]
            if extra:
                X = X.drop(columns=extra)

            X = X[self.feature_columns]


        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y, X.columns.tolist()

    def create_feature_importance_df(self, feature_names, importance_scores):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        return importance_df.sort_values('Importance', ascending=False)

    def save_preprocessors(self, path='models/', prefix=''):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(path, f'{prefix}scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(path, f'{prefix}label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(path, f'{prefix}feature_columns.pkl'))

        try:
            joblib.dump(self.column_defaults, os.path.join(path, f'{prefix}feature_defaults.pkl'))
        except Exception:
            pass

    def load_preprocessors(self, path='models/', prefix=''):
        self.scaler = joblib.load(os.path.join(path, f'{prefix}scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(path, f'{prefix}label_encoders.pkl'))
        self.feature_columns = joblib.load(os.path.join(path, f'{prefix}feature_columns.pkl'))

        defaults_path = os.path.join(path, f'{prefix}feature_defaults.pkl')
        if os.path.exists(defaults_path):
            try:
                self.column_defaults = joblib.load(defaults_path)
            except Exception:
                self.column_defaults = {}
        else:

            fallback = {}
            for col in (self.feature_columns or []):
                if col in self.label_encoders:
                    try:
                        fallback[col] = self.label_encoders[col].classes_[0]
                    except Exception:
                        fallback[col] = ''
                else:
                    fallback[col] = 0.0
            self.column_defaults = fallback
