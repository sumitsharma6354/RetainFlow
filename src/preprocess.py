import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Drop customerID
    df_customer_id = df['customerID']
    df = df.drop(columns=['customerID'])
    
    # Handle TotalCharges (empty strings to NaN, then median)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # SeniorCitizen as int (already 0/1 in source, but ensure)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    return df, df_customer_id

def get_preprocessor():
    # Binary columns (Yes/No) -> 1/0
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    # Service columns: "Yes"->1, others->0
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # One-hot encode: InternetService, Contract, PaymentMethod
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    # Numeric columns to scale
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Binary/Service mapping is not handled by ColumnTransformer easily if we want custom map.
    # We will handle mapping in load_and_clean or as part of a custom transformer.
    # Let's use a simpler approach: encode everything that isn't numeric as OHE or binary.
    
    return numeric_cols, ohe_cols + binary_cols + service_cols + ['gender']

def preprocess_df(df, is_training=True):
    # Map binary/service columns manually first
    mapping = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 'Male': 1, 'Female': 0}
    
    cols_to_map = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'gender']
    
    for col in cols_to_map:
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    
    # Churn mapping
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def build_pipeline(numeric_cols, ohe_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ohe_cols)
        ]
    )
    return preprocessor

if __name__ == "__main__":
    # Test script or manual run logic
    raw_path = "data/raw/telco_churn.csv"
    if os.path.exists(raw_path):
        df, ids = load_and_clean_data(raw_path)
        df = preprocess_df(df)
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
        
        # Binary ones are already numeric now
        
        preprocessor = build_pipeline(numeric_cols, ohe_cols)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_proc = preprocessor.fit_transform(X_train)
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_proc, y_train)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.joblib")
        print("Preprocessor saved to models/preprocessor.joblib")
