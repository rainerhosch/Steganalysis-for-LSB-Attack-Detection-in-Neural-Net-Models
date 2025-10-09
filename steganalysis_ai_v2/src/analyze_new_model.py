import torch
import argparse
import os
from src.config import config

def analyze_custom_model(model_path, model_type="auto"):
    """Analyze a custom model file"""
    
    # Load detector
    trained_model_path = os.path.join(config.MODEL_DIR, "trained", "best_stego_classifier.pkl")
    
    if not os.path.exists(trained_model_path):
        print("Trained model not found. Please train the model first.")
        return
    
    from predict import StegoDetector, load_model_from_path
    detector = StegoDetector(trained_model_path)
    
    # Determine model type
    if model_type == "auto":
        if "resnet50" in model_path.lower():
            model_type = "resnet50"
        elif "mobilenet" in model_path.lower():
            model_type = "mobilenet_v3_small"
        else:
            model_type = "resnet50"  # Default
    
    try:
        # Load the model
        model = load_model_from_path(model_path, model_type)
        model_name = os.path.basename(model_path)
        
        # Make prediction
        result = detector.predict_single_model(model, model_name)
        
        print("\n" + "="*60)
        print("MODEL ANALYSIS RESULT")
        print("="*60)
        print(f"Model: {result['model_name']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Stego Probability: {result['stego_probability']:.4f}")
        print(f"Clean Probability: {result['clean_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("="*60)
        
        # Interpretation
        if result['prediction'] == 'STEGO' and result['stego_probability'] > 0.7:
            print("🔍 HIGH CONFIDENCE: This model likely contains hidden payload!")
        elif result['prediction'] == 'CLEAN' and result['clean_probability'] > 0.7:
            print("✅ HIGH CONFIDENCE: This model appears to be clean.")
        else:
            print("⚠️  UNCERTAIN: The model shows ambiguous characteristics.")
            
    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a model for stego content')
    parser.add_argument('model_path', help='Path to the model file (.pth)')
    parser.add_argument('--model_type', default='auto', 
                    choices=['auto', 'resnet50', 'mobilenet_v3_small'],
                    help='Type of model architecture')
    
    args = parser.parse_args()
    
    analyze_custom_model(args.model_path, args.model_type)