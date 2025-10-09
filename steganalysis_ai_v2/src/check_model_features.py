import os
import joblib
import pandas as pd
from src.config import config

def check_trained_model(model="best_stego_classifier.pkl"):
    """Check what features the trained model expects"""
    model_path = os.path.join(config.MODEL_DIR, "trained", model)
    
    try:
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'feature_names' in model_data:
            print("✅ Model has feature names saved")
            feature_names = model_data['feature_names']
            model = model_data['model']
        else:
            print("❌ Model doesn't have feature names saved")
            model = model_data
            feature_names = None
        
        print(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
        
        if feature_names:
            print(f"Saved feature names ({len(feature_names)}):")
            for i, name in enumerate(feature_names):
                print(f"  {i+1:2d}. {name}")
        else:
            print("No feature names saved with model")
            
    except Exception as e:
        print(f"Error loading model: {e}")

def check_current_feature_extraction():
    """Check what features are currently being extracted"""
    import torchvision.models as models
    from src.feature_extraction_new import AdvancedFeatureExtractor, SteganalysisSpecificFeatures
    # from src.steganalysis_specific_features import SteganalysisSpecificFeatures
    
    # Test with a simple model
    model = models.resnet50(weights=None)
    
    # Extract features
    extractor = AdvancedFeatureExtractor()
    stego_extractor = SteganalysisSpecificFeatures()
    
    basic_features = extractor.extract_advanced_features(model, "test", "unknown")
    specific_features = stego_extractor.extract_lsb_specific_features(model)
    
    all_features = {**basic_features, **specific_features}
    
    print(f"Currently extracting {len(all_features)} features:")
    for i, (key, value) in enumerate(all_features.items()):
        print(f"  {i+1:2d}. {key}: {value}")
    
    return list(all_features.keys())

if __name__ == "__main__":
    print("=== CHECKING TRAINED MODEL ===")
    check_trained_model()
    
    print("\n=== CHECKING CURRENT FEATURE EXTRACTION ===")
    current_features = check_current_feature_extraction()
    
    print(f"\nTotal features currently extracted: {len(current_features)}")