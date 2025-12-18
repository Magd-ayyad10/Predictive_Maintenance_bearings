import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.config import data_path, model_save_path
from src.ingestion import load_dataset

def evaluate_model():
    print("🚀 Starting Model Evaluation...")

    # 1. Load the Trained Brain
    try:
        package = joblib.load(model_save_path)
        model = package['model']      # The PCA
        scaler = package['scaler']    # The Scaler
        threshold = package['threshold'] # The Red Line
        print(f"✅ Model loaded. Threshold is: {threshold:.4f}")
    except Exception as e:
        print("❌ Error: Model not found. Run train.py first.")
        return

    # 2. Load ALL Data (Healthy + Broken)
    print("📂 Loading full dataset (this takes time)...")
    df = load_dataset(data_path)

    # 3. Create "Ground Truth" Labels (The Trick)
    # Since NASA didn't label every row, we make a logical assumption:
    # - First 20% of life = Healthy (0)
    # - Last 5% of life = Broken (1)
    # - Middle = Grey area (We exclude it to be fair)
    
    labeled_data = []
    
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name]
        n = len(subset)
        
        # Indices for Healthy (Start) and Faulty (End)
        healthy_end = int(n * 0.20)
        fault_start = int(n * 0.95)
        
        # Healthy Chunk (Label 0)
        healthy_df = subset.iloc[:healthy_end].copy()
        healthy_df['y_true'] = 0
        
        # Faulty Chunk (Label 1)
        faulty_df = subset.iloc[fault_start:].copy()
        faulty_df['y_true'] = 1
        
        labeled_data.append(healthy_df)
        labeled_data.append(faulty_df)

    # Combine into a clean Test Set
    test_df = pd.concat(labeled_data)
    
    # 4. Prepare Data for Model
    X_test = test_df.drop(columns=['timestamp', 'dataset', 'y_true'])
    y_true = test_df['y_true']

    # 5. Make Predictions
    print("🧠 Running Predictions...")
    X_scaled = scaler.transform(X_test)
    
    # Calculate Reconstruction Error (MSE)
    recon = model.inverse_transform(model.transform(X_scaled))
    mse = np.mean(np.square(X_scaled - recon), axis=1)
    
    # If Error > Threshold, predict 1 (Anomaly), else 0 (Normal)
    y_pred = [1 if e > threshold else 0 for e in mse]

    # 6. Calculate Metrics
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANCE REPORT")
    print("="*40)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"✅ Accuracy:  {acc:.2%}")
    print(f"🎯 Precision: {prec:.2%} (When we say it's broken, is it really?)")
    print(f"🔍 Recall:    {rec:.2%} (Did we catch all the failures?)")
    print(f"⚖️ F1 Score:  {f1:.2%} (Balance between Precision & Recall)")
    
    print("\nDetailed Breakdown:")
    print(confusion_matrix(y_true, y_pred))
    print("\n[True Normal, False Alarm]")
    print("[Missed Fault, True Detection]")

if __name__ == "__main__":
    evaluate_model()