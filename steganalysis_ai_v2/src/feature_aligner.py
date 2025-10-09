import pandas as pd
import numpy as np
import joblib
import os
from src.config import config

class FeatureAligner:
    def __init__(self, model_path):
        """Align features with trained model"""
        self.model_data = joblib.load(model_path)
        self.classifier = self._extract_classifier(self.model_data)
        self.expected_features = self._get_expected_features()
        
        print(f"📊 Model expects {self.expected_features} features")
    
    def _extract_classifier(self, model_data):
        """Extract classifier from model data"""
        if hasattr(model_data, 'predict'):
            return model_data
        elif isinstance(model_data, dict):
            for key in ['classifier', 'model', 'best_model']:
                if key in model_data and hasattr(model_data[key], 'predict'):
                    return model_data[key]
        return model_data
    
    def _get_expected_features(self):
        """Get number of features expected by the model"""
        if hasattr(self.classifier, 'n_features_in_'):
            return self.classifier.n_features_in_
        elif isinstance(self.model_data, dict) and 'feature_names' in self.model_data:
            return len(self.model_data['feature_names'])
        else:
            # Default fallback
            return 12
    
    def align_features(self, extracted_features):
        """Align extracted features to match model expectations"""
        if hasattr(self.classifier, 'feature_names_in_') or \
        (isinstance(self.model_data, dict) and 'feature_names' in self.model_data):
            # Model was trained with feature names - use exact alignment
            return self._align_with_feature_names(extracted_features)
        else:
            # Model was trained without feature names - use positional alignment
            return self._align_positional(extracted_features)
    
    def _align_with_feature_names(self, extracted_features):
        """Align using feature names from training"""
        if isinstance(self.model_data, dict) and 'feature_names' in self.model_data:
            feature_names = self.model_data['feature_names']
        else:
            feature_names = self.classifier.feature_names_in_
        
        aligned_features = {}
        
        for feature_name in feature_names:
            if feature_name in extracted_features:
                aligned_features[feature_name] = extracted_features[feature_name]
            else:
                print(f"⚠️  Feature '{feature_name}' not found, using 0")
                aligned_features[feature_name] = 0.0
        
        # Create DataFrame with correct column order
        features_df = pd.DataFrame([aligned_features])[feature_names]
        return features_df
    
    def _align_positional(self, extracted_features):
        """Align features positionally (first N features)"""
        # Convert to list of (feature_name, value) pairs
        feature_list = [(k, v) for k, v in extracted_features.items() 
                    if isinstance(v, (int, float)) and not np.isnan(v)]
        
        # Sort by feature name for consistency
        feature_list.sort(key=lambda x: x[0])
        
        # Take first N features
        selected_features = feature_list[:self.expected_features]
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(self.expected_features)]
        feature_values = [val for _, val in selected_features]
        
        # Pad with zeros if needed
        while len(feature_values) < self.expected_features:
            feature_values.append(0.0)
        
        return pd.DataFrame([feature_values], columns=feature_names)

# Standard feature set that matches most models
STANDARD_FEATURES = [
    'weight_mean', 'weight_std', 'weight_skew', 'weight_kurtosis',
    'reconstruction_loss', 'grad_mean', 'grad_std', 'grad_max',
    'entropy_bit_0', 'entropy_bit_1', 'entropy_bit_2', 'entropy_bit_3'
]