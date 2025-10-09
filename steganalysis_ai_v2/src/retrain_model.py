import pandas as pd
import os
from src.classifier import StegoClassifier
from src.feature_selection import PSOFeatureSelection
from src.config import config

def retrain_with_consistent_features(datasets="all_features.csv", conf="normal"):
    """Retrain model with consistent feature set"""
    
    # Load features
    features_path = os.path.join(config.DATA_DIR, "features", datasets)
    if not os.path.exists(features_path):
        print(f"Features file not found: {features_path}")
        return
    
    features_df = pd.read_csv(features_path)
    print(f"Loaded features: {features_df.shape}")
    
    # Select only the most important features for consistency
    important_features = [
        'weight_mean', 'weight_std', 'weight_skew', 'weight_kurtosis',
        'reconstruction_loss', 'grad_mean', 'grad_std', 'grad_max',
        'entropy_bit_0', 'entropy_bit_1', 'entropy_bit_2', 'entropy_bit_3',
        'weight_median', 'weight_near_zero', 'bit_transition_mean', 'spatial_corr_mean'
    ]
    
    # Filter features to only include available ones
    available_features = [f for f in important_features if f in features_df.columns]
    print(f"Using {len(available_features)} consistent features: {available_features}")
    
    # Prepare data with consistent features
    classifier = StegoClassifier()
    
    # Use only the selected features
    selected_features_df = features_df[available_features + ['model_name', 'model_type', 'is_stego']]
    
    # Train without feature selection (use all selected features)
    # Method now returns 3 values: results, best_model, feature_names
    results, best_model, feature_names = classifier.train_and_evaluate_v2(selected_features_df, feature_mask=None)
    
    # Save model with feature names, choose filename based on conf
    if conf == "normal":
        filename = "consistent_stego_classifier.pkl"
    else:
        filename = f"consistent_stego_classifier_{conf}.pkl"
    
    classifier.save_model_v2(best_model, feature_names, filename)
    
    print(f"Model trained and saved as: {filename}")
    print(f"Features used: {len(feature_names)}")
    
    return best_model, feature_names, results

if __name__ == "__main__":
    retrain_with_consistent_features()