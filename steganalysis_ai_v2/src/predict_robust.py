import torch
import pandas as pd
import joblib
import os
import numpy as np
from src.config import config
from src.feature_aligner import FeatureAligner, STANDARD_FEATURES

class RobustStegoDetector:
    def __init__(self, model_path):
        """Robust detector with feature alignment"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize feature aligner
        self.aligner = FeatureAligner(model_path)
        self.classifier = self.aligner.classifier
        
        from src.feature_extraction_new import AdvancedFeatureExtractor, SteganalysisSpecificFeatures
        # from src.steganalysis_specific_features import SteganalysisSpecificFeatures
        
        self.feature_extractor = AdvancedFeatureExtractor()
        self.stego_extractor = SteganalysisSpecificFeatures()
        
        print(f"🤖 Classifier: {type(self.classifier).__name__}")
        print(f"📊 Expected features: {self.aligner.expected_features}")
    
    def extract_and_align_features(self, model, model_name="unknown"):
        """Extract and align features for prediction"""
        try:
            # Extract all features
            basic_features = self.feature_extractor.extract_advanced_features(model, model_name, "unknown")
            specific_features = self.stego_extractor.extract_lsb_specific_features(model)
            all_features = {**basic_features, **specific_features}
            
            # Remove non-numeric and metadata
            numeric_features = {}
            for key, value in all_features.items():
                if (isinstance(value, (int, float)) and 
                    not np.isnan(value) and 
                    not np.isinf(value) and
                    key not in ['model_name', 'model_type', 'is_stego']):
                    numeric_features[key] = value
            
            print(f"📈 Extracted {len(numeric_features)} numeric features")
            
            # Align with model expectations
            aligned_features = self.aligner.align_features(numeric_features)
            
            print(f"✅ Aligned to {aligned_features.shape[1]} features for prediction")
            return aligned_features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # Return properly aligned dummy features
            return self._create_dummy_features()
    
    def _create_dummy_features(self):
        """Create dummy features with correct dimensions"""
        if hasattr(self.aligner, 'expected_features'):
            n_features = self.aligner.expected_features
            feature_names = [f"feature_{i}" for i in range(n_features)]
            return pd.DataFrame([[0.0] * n_features], columns=feature_names)
        else:
            return pd.DataFrame([[0.0] * 12])  # Default fallback
    
    def predict_single(self, model, model_name="unknown"):
        """Predict a single model"""
        try:
            features = self.extract_and_align_features(model, model_name)
            
            # Verify dimensions
            if features.shape[1] != self.aligner.expected_features:
                raise ValueError(f"Feature mismatch: got {features.shape[1]}, expected {self.aligner.expected_features}")
            
            # Make prediction
            prediction = self.classifier.predict(features)[0]
            
            # Get probabilities
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(features)[0]
                stego_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                clean_prob = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
                confidence = max(probabilities)
            else:
                stego_prob = 1.0 if prediction == 1 else 0.0
                clean_prob = 1.0 - stego_prob
                confidence = 1.0
            
            result = {
                'prediction': 'STEGO' if prediction == 1 else 'CLEAN',
                'confidence': float(confidence),
                'stego_probability': float(stego_prob),
                'clean_probability': float(clean_prob),
                'model_name': model_name,
                'features_used': features.shape[1]
            }
            
            print(f"✅ {model_name}: {result['prediction']} "
                f"(Stego: {result['stego_probability']:.3f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction failed for {model_name}: {e}")
            return {
                'prediction': 'ERROR',
                'error': str(e),
                'model_name': model_name
            }
    
    def predict_batch(self, models_dict):
        """Predict multiple models"""
        results = []
        
        print(f"🔍 Analyzing {len(models_dict)} models...")
        for model_name, model in models_dict.items():
            result = self.predict_single(model, model_name)
            results.append(result)
        
        return pd.DataFrame(results)

def test_robust_detector():
    """Test the robust detector"""
    model_path = os.path.join(config.MODEL_DIR, "trained", "best_stego_classifier.pkl")
    
    # Try alternative paths
    if not os.path.exists(model_path):
        alternative_paths = [
            os.path.join(config.MODEL_DIR, "trained", "working_classifier.pkl"),
            os.path.join(config.MODEL_DIR, "trained", "consistent_stego_classifier.pkl"),
            os.path.join(config.MODEL_DIR, "trained", "simple_consistent_classifier.pkl"),
            os.path.join(config.MODEL_DIR, "trained", "emergency_model.pkl"),
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"📁 Using alternative model: {alt_path}")
                break
        else:
            print("❌ No model file found. Please train a model first.")
            return
    
    try:
        detector = RobustStegoDetector(model_path)
        
        # Test with sample models
        import torchvision.models as models
        test_models = {
            "resnet50_clean": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            "mobilenet_clean": models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        }
        
        results = detector.predict_batch(test_models)
        
        # Display results
        print(f"\n{'='*60}")
        print("📊 PREDICTION RESULTS")
        print(f"{'='*60}")
        
        for _, row in results.iterrows():
            status_icon = "🔴" if row['prediction'] == 'STEGO' else "🟢" if row['prediction'] == 'CLEAN' else "❌"
            print(f"{status_icon} {row['model_name']:25} -> {row['prediction']:8} "
                f"(Stego: {row['stego_probability']:.3f}, Conf: {row['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")

if __name__ == "__main__":
    test_robust_detector()