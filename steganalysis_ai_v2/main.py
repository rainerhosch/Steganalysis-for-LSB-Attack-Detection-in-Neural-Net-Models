import pandas as pd
import os
from src.model_acquisition import ModelAcquisition
from src.stego_generator import StegoGenerator
from src.feature_extraction import FeatureExtractor
from src.feature_selection import PSOFeatureSelection
from src.classifier import StegoClassifier
from src.config import config

def main():
    print("Starting AI Model Steganalysis Research...")
    
    # Step 1: Model Acquisition
    print("\n=== Step 1: Model Acquisition ===")
    model_acquirer = ModelAcquisition()
    original_models = model_acquirer.download_pretrained_models()
    
    # Step 2: Stego Model Creation
    print("\n=== Step 2: Stego Model Creation ===")
    injector = StegoGenerator()
    stego_models = injector.create_stego_models(original_models)
    
    # Step 3: Feature Extraction
    print("\n=== Step 3: Feature Extraction ===")
    feature_extractor = FeatureExtractor()
    
    all_features = []
    
    # Extract features from cover models
    for model_name, model in original_models.items():
        features = feature_extractor.extract_all_features(model, model_name, "cover")
        all_features.append(features)
    
    # Extract features from stego models
    for model_name, model in stego_models.items():
        features = feature_extractor.extract_all_features(model, model_name, "stego")
        all_features.append(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(all_features)
    features_path = os.path.join(config.DATA_DIR, "features", "all_features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")
    
    # Step 4: Feature Selection with PSO
    print("\n=== Step 4: Feature Selection with PSO ===")
    classifier = StegoClassifier()
    X, y = classifier.prepare_data(features_df)
    
    pso_selector = PSOFeatureSelection()
    feature_mask, best_score = pso_selector.optimize(X, y)
    
    # Step 5: Model Training and Evaluation
    print("\n=== Step 5: Model Training and Evaluation ===")
    results, best_model = classifier.train_and_evaluate(features_df, feature_mask)
    
    # Save the best model
    classifier.save_model(best_model)
    
    print("\n=== Research Completed ===")
    print("Summary of Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: F1-Score = {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()