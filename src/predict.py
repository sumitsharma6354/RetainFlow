import joblib
import pandas as pd
import numpy as np
import shap
import os
from src.preprocess import preprocess_df

# Load models and preprocessor globally to avoid reloading on every request
MODELS_PATH = "models/"
preprocessor = None
xgb_model = None
kmeans_model = None
explainer = None

def load_models():
    global preprocessor, xgb_model, kmeans_model, explainer
    if preprocessor is None:
        preprocessor = joblib.load(os.path.join(MODELS_PATH, "preprocessor.joblib"))
    if xgb_model is None:
        xgb_model = joblib.load(os.path.join(MODELS_PATH, "xgb_model.joblib"))
    if kmeans_model is None:
        kmeans_model = joblib.load(os.path.join(MODELS_PATH, "kmeans_model.joblib"))
    if explainer is None:
        explainer = shap.TreeExplainer(xgb_model)

def predict_customer(customer_features: dict, include_shap: bool = True) -> dict:
    load_models()
    
    # Create DataFrame
    df = pd.DataFrame([customer_features])
    
    # Preprocess
    df_proc_mapped = preprocess_df(df.copy())
    
    # Segment (using original features)
    segment_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # TotalCharges might be string in input
    df_proc_mapped['TotalCharges'] = pd.to_numeric(df_proc_mapped['TotalCharges'], errors='coerce').fillna(0)
    
    segment_id = kmeans_model.predict(df_proc_mapped[segment_features])[0]
    
    # Mapping for segment names (should match src/segment.py)
    centers = kmeans_model.cluster_centers_
    t, m, tot = centers[segment_id]
    if t > 30 and m > 60: segment_name = "High-value loyal"
    elif t <= 30 and m > 60: segment_name = "New at-risk"
    elif t > 30 and m <= 60: segment_name = "Long-term stable"
    else: segment_name = "Occasional user"
    
    # Process for XGB
    X_proc = preprocessor.transform(df_proc_mapped)
    
    # Predict
    churn_prob = float(xgb_model.predict_proba(X_proc)[0][1])
    churn_pred = bool(churn_prob >= 0.5)
    
    # Risk Level
    if churn_prob >= 0.7: risk_level = "High"
    elif churn_prob >= 0.4: risk_level = "Medium"
    else: risk_level = "Low"
    
    shap_data = []
    if include_shap:
        # SHAP
        shap_values = explainer.shap_values(X_proc)[0]
        
        # Feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()
        
        for i, val in enumerate(shap_values):
            shap_data.append({
                "feature": feature_names[i],
                "value": float(val),
                "direction": "increases_risk" if val > 0 else "decreases_risk"
            })
        
        # Sort by absolute value and take top 6
        shap_data = sorted(shap_data, key=lambda x: abs(x['value']), reverse=True)[:6]
        
        # Clean feature names for UI
        for item in shap_data:
            name = item['feature']
            if "cat__" in name: name = name.replace("cat__", "")
            if "num__" in name: name = name.replace("num__", "")
            item['feature'] = name.replace("_", " ").title()

    return {
        "churn_probability": churn_prob,
        "churn_prediction": churn_pred,
        "risk_level": risk_level,
        "segment": segment_name,
        "shap_values": shap_data
    }
