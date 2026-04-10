import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_models(X_train_res, y_train_res, X_test_proc, y_test):
    # XGBoost
    # scale_pos_weight = (count of non-churn / count of churn)
    # Since we used SMOTE, classes are balanced in resampled set, but user requested specific params
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    xgb_model.fit(X_train_res, y_train_res)
    
    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train_res, y_train_res)
    
    # Evaluate XGB
    y_pred = xgb_model.predict(X_test_proc)
    y_prob = xgb_model.predict_proba(X_test_proc)[:, 1]
    
    print("XGBoost Evaluation:")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/xgb_model.joblib")
    joblib.dump(lr_model, "models/lr_model.joblib")
    
    return xgb_model, lr_model
