import numpy as np
import pandas as pd
import os
from src.model_acquisition import ModelAcquisition
from src.stego_generator import StegoGenerator
from src.feature_extractor import FeatureExtractor
from src.pso_optimizer import PSOFeatureSelector
from src.classifier import StegoClassifier
from src.config import Config

def main():
    config = Config()
    
    print("=== AI STEGANALYSIS RESEARCH PIPELINE ===")
    
    # 1. Akuisisi Model
    print("\n1. Downloading pre-trained models...")
    model_acq = ModelAcquisition()
    clean_models = model_acq.download_pytorch_models()
    
    # Simpan model cover
    for model_name, model in clean_models.items():
        model_acq.save_model_weights(model, model_name)
    
    # 2. Generate Stego Models
    print("\n2. Generating stego models...")
    stego_gen = StegoGenerator()
    stego_models = stego_gen.create_stego_models(clean_models)
    
    # 3. Ekstraksi Fitur
    print("\n3. Extracting features...")
    feature_extractor = FeatureExtractor()
    
    cover_features = []
    stego_features = []
    
    # Ekstrak fitur model cover
    for model_name, model in clean_models.items():
        features = feature_extractor.extract_all_features(model.state_dict())
        cover_features.append(list(features.values()))
        print(f"Extracted features from cover model: {model_name}")
    
    # Ekstrak fitur model stego
    for model_name, stego_variants in stego_models.items():
        for variant_name, weights in stego_variants.items():
            features = feature_extractor.extract_all_features(weights)
            stego_features.append(list(features.values()))
            print(f"Extracted features from stego model: {variant_name}")
    
    # 4. Persiapan Dataset
    print("\n4. Preparing dataset...")
    classifier = StegoClassifier()
    X, y = classifier.prepare_dataset(cover_features, stego_features)
    
    # Simpan fitur
    feature_df = pd.DataFrame(X)
    feature_df['label'] = y
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    feature_df.to_csv(f"{config.RESULTS_PATH}extracted_features.csv", index=False)
    
    # 5. Optimisasi Fitur dengan PSO
    print("\n5. Optimizing features with PSO...")
    pso_selector = PSOFeatureSelector(X, y)
    best_features, best_score = pso_optimizer.optimize()
    
    print(f"Best F1-Score: {best_score:.4f}")
    print(f"Selected {np.sum(best_features)} out of {len(best_features)} features")
    
    # 6. Training dan Evaluasi
    print("\n6. Training and evaluating classifiers...")
    results = classifier.train_and_evaluate(X, y, best_features)
    
    # 7. Simpan Hasil
    print("\n7. Saving results...")
    classifier.save_models(results, f"{config.RESULTS_PATH}models/")
    
    # Simpan metrics
    metrics_df = pd.DataFrame({
        model_name: result['metrics'] 
        for model_name, result in results.items()
    }).T
    
    metrics_df.to_csv(f"{config.RESULTS_PATH}evaluation_metrics.csv")
    print("\n=== PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main()