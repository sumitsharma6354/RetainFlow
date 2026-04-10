import pandas as pd
import numpy as np
import joblib
import os
import time
from sqlalchemy.orm import Session
from database.db import engine, init_db, SessionLocal, Customer, Prediction, RetentionAction
from src.predict import predict_customer, load_models
from src.retention import get_retention_action
from datetime import datetime

def seed_db():
    start_time = time.time()
    print("Starting robust database seeding...")
    
    # 1. Initialize DB
    init_db()
    db = SessionLocal()
    
    # 2. Check source data
    raw_path = "data/raw/telco_churn.csv"
    if not os.path.exists(raw_path):
        print(f"Error: Dataset not found at {raw_path}")
        return

    # 3. Check models
    model_files = ["xgb_model.joblib", "preprocessor.joblib", "kmeans_model.joblib"]
    for f in model_files:
        if not os.path.exists(f"models/{f}"):
            print(f"Warning: {f} not found in /models. Running train_pipeline.py...")
            import subprocess
            subprocess.run(["python", "train_pipeline.py"], check=True)
            break
            
    # Load models
    load_models()
    
    # 4. Clean existing data if any? 
    if db.query(Customer).count() > 0:
        print("Database already contains data. Skipping seeding.")
        db.close()
        return

    df = pd.read_csv(raw_path)
    total_rows = len(df)
    print(f"Found {total_rows} customers in CSV.")

    customers = []
    predictions = []
    retention_actions = []

    try:
        for i, row in df.iterrows():
            customer_id = row['customerID']
            
            # Create Customer record
            customer = Customer(
                customer_id=customer_id,
                gender=row['gender'],
                senior_citizen=int(row['SeniorCitizen']),
                partner=row['Partner'],
                dependents=row['Dependents'],
                tenure=int(row['tenure']),
                phone_service=row['PhoneService'],
                multiple_lines=row['MultipleLines'],
                internet_service=row['InternetService'],
                online_security=row['OnlineSecurity'],
                online_backup=row['OnlineBackup'],
                device_protection=row['DeviceProtection'],
                tech_support=row['TechSupport'],
                streaming_tv=row['StreamingTV'],
                streaming_movies=row['StreamingMovies'],
                contract=row['Contract'],
                paperless_billing=row['PaperlessBilling'],
                payment_method=row['PaymentMethod'],
                monthly_charges=float(row['MonthlyCharges']),
                total_charges=pd.to_numeric(row['TotalCharges'], errors='coerce') if not pd.isna(row['TotalCharges']) else 0.0,
                churn_actual=1 if row['Churn'] == 'Yes' else 0
            )
            customers.append(customer)
            
            feat_dict = row.to_dict()
            
            # Extract features for XGB
            from src.predict import preprocessor, xgb_model, kmeans_model, preprocess_df
            
            # Manual prediction logic to bypass SHAP
            df_row = pd.DataFrame([feat_dict])
            df_proc = preprocess_df(df_row.copy())
            
            # Segment
            seg_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'], errors='coerce').fillna(0)
            segment_id = kmeans_model.predict(df_proc[seg_features])[0]
            
            centers = kmeans_model.cluster_centers_
            t, m, tot = centers[segment_id]
            if t > 30 and m > 60: segment_name = "High-value loyal"
            elif t <= 30 and m > 60: segment_name = "New at-risk"
            elif t > 30 and m <= 60: segment_name = "Long-term stable"
            else: segment_name = "Occasional user"
            
            # XGB Prob
            X_proc = preprocessor.transform(df_proc)
            churn_prob = float(xgb_model.predict_proba(X_proc)[0][1])
            churn_pred = bool(churn_prob >= 0.5)
            
            if churn_prob >= 0.7: risk_level = "High"
            elif churn_prob >= 0.4: risk_level = "Medium"
            else: risk_level = "Low"
            
            pred = Prediction(
                customer_id=customer_id,
                churn_probability=churn_prob,
                churn_prediction=churn_pred,
                risk_level=risk_level,
                segment=segment_name,
                predicted_at=datetime.utcnow()
            )
            predictions.append(pred)
            
            # Retention
            ret_res = get_retention_action(feat_dict, churn_prob, segment_name)
            ret = RetentionAction(
                customer_id=customer_id,
                offer=ret_res['offer'],
                urgency=ret_res['urgency'],
                status="to_contact",
                estimated_save_value=ret_res['estimated_save_value'],
                created_at=datetime.utcnow()
            )
            retention_actions.append(ret)
            
            # Progress and Batching
            if (i + 1) % 500 == 0:
                db.bulk_save_objects(customers)
                db.bulk_save_objects(predictions)
                db.bulk_save_objects(retention_actions)
                db.commit()
                customers, predictions, retention_actions = [], [], []
                print(f"Seeded {i + 1}/{total_rows}...")

        # Final batch
        if customers:
            db.bulk_save_objects(customers)
            db.bulk_save_objects(predictions)
            db.bulk_save_objects(retention_actions)
            db.commit()
            print(f"Seeded {total_rows}/{total_rows}...")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        
    end_time = time.time()
    print(f"Seeding complete! Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    seed_db()
