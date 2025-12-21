import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.config import data_path, model_save_path, train_split_ratio, pca_components, threshold_buffer
from src.ingestion import load_dataset


def train_pipeline():
    # 1. Ingestion
    df = load_dataset(data_path)
    
    # 2. Split Healthy Data
    training_pieces = []
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name]
        limit = int(len(subset) * train_split_ratio)
        training_pieces.append(subset.iloc[:limit])
    
    train_df = pd.concat(training_pieces).drop(columns=['dataset'])
    print(f"Training on {len(train_df)} healthy rows.")

    # 3. Pipeline Construction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df)
    
    pca = PCA(n_components=pca_components)
    pca.fit(X_scaled)
    
    # 4. Threshold Calculation
    recon = pca.inverse_transform(pca.transform(X_scaled))
    mse = np.mean(np.square(X_scaled - recon), axis=1)
    threshold = np.max(mse) * threshold_buffer
    
    print(f"Model Trained. Threshold: {threshold:.4f}")

    # 5. Save Artifacts
    package = {
        'model': pca,
        'scaler': scaler,
        'threshold': threshold
    }
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    joblib.dump(package, model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    train_pipeline()