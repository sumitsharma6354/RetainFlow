from src.preprocess import load_and_clean_data, preprocess_df, build_pipeline
from src.segment import segment_customers
from src.train import train_models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import pandas as pd

def run_pipeline():
    print("Starting RetainFlow Training Pipeline...")
    
    raw_path = "data/raw/telco_churn.csv"
    if not os.path.exists(raw_path):
        print(f"Error: Dataset not found at {raw_path}")
        return
    
    # 1. Load and clean
    df, ids = load_and_clean_data(raw_path)
    
    # 2. Segment (before preprocessing for SMOTE, but on original features)
    df, kmeans = segment_customers(df)
    joblib.dump(kmeans, "models/kmeans_model.joblib")
    print("Segmentation complete.")
    
    # 3. Preprocess features
    df = preprocess_df(df)
    
    X = df.drop(['Churn', 'segment', 'segment_id'], axis=1) # Don't use segment ID as feature? User didn't specify, usually you don't.
    y = df['Churn']
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    preprocessor = build_pipeline(numeric_cols, ohe_cols)
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Fit Preprocessor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print("Preprocessing complete and saved.")
    
    # 6. Apply SMOTE on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    print("Class balancing (SMOTE) complete.")
    
    # 7. Train
    train_models(X_train_res, y_train_res, X_test_proc, y_test)
    
    print("\nTraining complete. Models saved to /models")

if __name__ == "__main__":
    run_pipeline()
