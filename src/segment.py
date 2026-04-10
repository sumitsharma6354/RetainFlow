import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

def segment_customers(df, n_clusters=4):
    # Use: tenure, MonthlyCharges, TotalCharges
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[features]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['segment_id'] = kmeans.fit_transform(X).argmin(axis=1) # This is just a placeholder logic for cluster labels
    # Actually fit and then predict
    kmeans.fit(X)
    df['segment_id'] = kmeans.labels_
    
    # Assign names based on cluster centers
    centers = kmeans.cluster_centers_
    # centers columns: 0: tenure, 1: MonthlyCharges, 2: TotalCharges
    
    # Logic to label
    # High-value loyal (high tenure, high charges)
    # New at-risk (low tenure, high charges)
    # Long-term stable (high tenure, low charges)
    # Occasional user (low tenure, low charges)
    
    segment_names = {}
    for i, center in enumerate(centers):
        tenure, monthly, total = center
        if tenure > 30 and monthly > 60:
            segment_names[i] = "High-value loyal"
        elif tenure <= 30 and monthly > 60:
            segment_names[i] = "New at-risk"
        elif tenure > 30 and monthly <= 60:
            segment_names[i] = "Long-term stable"
        else:
            segment_names[i] = "Occasional user"
            
    df['segment'] = df['segment_id'].map(segment_names)
    
    return df, kmeans

if __name__ == "__main__":
    # Test/Train
    from src.preprocess import load_and_clean_data
    df, ids = load_and_clean_data("data/raw/telco_churn.csv")
    df, kmeans = segment_customers(df)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.joblib")
    print("KMeans model saved to models/kmeans_model.joblib")
    print(df[['segment_id', 'segment']].value_counts())
