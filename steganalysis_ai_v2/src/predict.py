import torch
import torchvision.models as models
import pandas as pd
import numpy as np
import joblib
import os
from src.feature_extraction_new import AdvancedFeatureExtractor, SteganalysisSpecificFeatures
# from steganalysis_specific_features import SteganalysisSpecificFeatures
from src.config import config

class StegoDetector:
    def __init__(self, model_path):
        """Initialize the stego detector with trained model"""
        self.model = joblib.load(model_path)
        self.feature_extractor = AdvancedFeatureExtractor()
        self.stego_extractor = SteganalysisSpecificFeatures()
        print(f"Loaded model: {type(self.model).__name__}")
        
    def extract_features_from_model(self, model, model_name="unknown"):
        """Extract features from a single model for prediction"""
        # Extract basic features
        basic_features = self.feature_extractor.extract_advanced_features(model, model_name, "unknown")
        
        # Extract stego-specific features
        specific_features = self.stego_extractor.extract_lsb_specific_features(model)
        
        # Combine all features
        all_features = {**basic_features, **specific_features}
        
        # Convert to DataFrame
        features_df = pd.DataFrame([all_features])
        
        # Remove non-numeric columns
        feature_columns = [col for col in features_df.columns 
                        if col not in ['model_name', 'model_type', 'is_stego'] 
                        and pd.api.types.is_numeric_dtype(features_df[col])]
        
        return features_df[feature_columns]
    
    def predict_single_model_1(self, model, model_name="unknown"):
        """Predict if a single model contains stego payload"""
        try:
            # Extract features
            features = self.extract_features_from_model(model, model_name)
            
            # Handle missing features (fill with mean/median if needed)
            features = features.fillna(0)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            return {
                'prediction': 'STEGO' if prediction == 1 else 'CLEAN',
                'confidence': max(probability),
                'stego_probability': probability[1],
                'clean_probability': probability[0],
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"Error predicting model {model_name}: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'stego_probability': 0.0,
                'clean_probability': 0.0,
                'model_name': model_name,
                'error': str(e)
            }
    
    def predict_single_model(self, model, model_name="unknown"):
        """Predict if a single model contains stego payload"""
        try:
            # Extract and align features
            features = self.extract_features_from_model(model, model_name)
            
            # # Verify feature dimensions
            # if features.shape[1] != len(self.training_feature_names):
            #     raise ValueError(f"Feature dimension mismatch: got {features.shape[1]}, expected {len(self.training_feature_names)}")
            
            # Handle missing values
            features = features.fillna(0)
            
            # **FIX: Access the actual classifier model, not the dictionary**
            if isinstance(self.model, dict) and 'classifier' in self.model:
                classifier = self.model['classifier']
            elif isinstance(self.model, dict) and 'model' in self.model:
                classifier = self.model['model']
            else:
                classifier = self.model  # Assume it's already the classifier
            
            # Make prediction using the actual classifier
            prediction = classifier.predict(features)[0]
            probability = classifier.predict_proba(features)[0]
            
            return {
                'prediction': 'STEGO' if prediction == 1 else 'CLEAN',
                'confidence': max(probability),
                'stego_probability': probability[1],
                'clean_probability': probability[0],
                'model_name': model_name,
                'features_used': len(self.training_feature_names)
            }
            
        except Exception as e:
            print(f"Error predicting model {model_name}: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'stego_probability': 0.0,
                'clean_probability': 0.0,
                'model_name': model_name,
                'error': str(e)
            }

    def predict_multiple_models(self, models_dict):
        """Predict multiple models at once"""
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Analyzing: {model_name}")
            result = self.predict_single_model(model, model_name)
            results.append(result)
            
            # Print immediate result
            print(f"  Result: {result['prediction']} "
                f"(Confidence: {result['confidence']:.3f}, "
                f"Stego Prob: {result['stego_probability']:.3f})")
        
        return pd.DataFrame(results)

def load_model_from_path(model_path, model_name="resnet50"):
    """Load a PyTorch model from file path"""
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def main():
    # Initialize detector
    model_path = os.path.join(config.MODEL_DIR, "trained", "best_stego_classifier.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    detector = StegoDetector(model_path)
    
    # Test with some models
    test_models = {}
    
    # Load cover models for testing
    print("Loading test models...")
    
    # Example: Load original cover models
    from src.model_acquisition import ModelAcquisition
    acquirer = ModelAcquisition()
    original_models = acquirer.download_pretrained_models()
    
    # Add cover models to test
    for name, model in original_models.items():
        test_models[f"cover_{name}"] = model
    
    # Load some stego models for testing
    stego_dir = os.path.join(config.MODEL_DIR, "stego")
    if os.path.exists(stego_dir):
        stego_files = [f for f in os.listdir(stego_dir) if f.endswith('.pth')]
        
        for stego_file in stego_files[:3]:  # Test first 3 stego models
            model_path = os.path.join(stego_dir, stego_file)
            try:
                if "resnet50" in stego_file:
                    model = load_model_from_path(model_path, "resnet50")
                    test_models[f"stego_{stego_file}"] = model
                elif "mobilenet" in stego_file:
                    model = load_model_from_path(model_path, "mobilenet_v3_small")
                    test_models[f"stego_{stego_file}"] = model
            except Exception as e:
                print(f"Failed to load {stego_file}: {e}")
    
    # Make predictions
    print(f"\nAnalyzing {len(test_models)} models...")
    results = detector.predict_multiple_models(test_models)
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    for _, row in results.iterrows():
        status = "✅" if row['prediction'] in ['CLEAN', 'STEGO'] else "❌"
        print(f"{status} {row['model_name']:50} -> {row['prediction']:8} "
            f"(Stego: {row['stego_probability']:.3f}, "
            f"Confidence: {row['confidence']:.3f})")
    
    # Summary statistics
    clean_count = len(results[results['prediction'] == 'CLEAN'])
    stego_count = len(results[results['prediction'] == 'STEGO'])
    error_count = len(results[results['prediction'] == 'ERROR'])
    
    print(f"\nSUMMARY:")
    print(f"Clean models: {clean_count}")
    print(f"Stego models: {stego_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()