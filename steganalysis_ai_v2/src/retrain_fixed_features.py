import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import config
from src.feature_aligner import FeatureAligner, STANDARD_FEATURES

def retrain_with_fixed_features(datasets="all_features.csv", mode=""):
    """Retrain model with fixed, consistent feature set"""
    
    # Load features
    features_path = os.path.join(config.DATA_DIR, "features", datasets)
    if not os.path.exists(features_path):
        print("❌ Features file not found")
        return
    
    features_df = pd.read_csv(features_path)
    print(f"📁 Loaded features: {features_df.shape}")
    
    # Use ONLY the standard 12 features
    fixed_features = [f for f in STANDARD_FEATURES if f in features_df.columns]
    
    # If some features are missing, create them with default values
    for feature in STANDARD_FEATURES:
        if feature not in features_df.columns:
            print(f"⚠️  Creating missing feature: {feature}")
            if 'entropy' in feature:
                features_df[feature] = 0.5  # Default entropy
            elif 'weight' in feature:
                features_df[feature] = 0.0  # Default weight stats
            elif 'grad' in feature:
                features_df[feature] = 0.0  # Default gradient
            else:
                features_df[feature] = 0.0
    
    print(f"📊 Using {len(fixed_features)} fixed features")
    print(f"Features: {fixed_features}")
    
    # Prepare data
    X = features_df[fixed_features]
    y = features_df['is_stego']
    
    # Handle NaN values
    X = X.fillna(0)
    
    print(f"🎯 Final dataset: {X.shape}")
    print(f"📈 Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save model with feature names
    model_data = {
        'classifier': model,
        'feature_names': fixed_features,
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'feature_set': 'standard_12_features'
    }
    
    model_path = os.path.join(config.MODEL_DIR, "trained", f"fixed_features_classifier_{mode}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"📊 Features: {len(fixed_features)}")
    print(f"🔧 Feature set: standard_12_features")
    
    return model, fixed_features

if __name__ == "__main__":
    retrain_with_fixed_features()