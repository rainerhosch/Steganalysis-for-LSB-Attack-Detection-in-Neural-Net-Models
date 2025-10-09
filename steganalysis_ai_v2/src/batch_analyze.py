import os
import pandas as pd
from tqdm import tqdm
from src.config import config

def batch_analyze_models(models_directory, predict_models="consistent_stego_classifier.pkl"):
    """Analyze all models in a directory"""
    
    # Load detector
    trained_model_path = os.path.join(config.MODEL_DIR, "trained", predict_models)
    # trained_model_path = os.path.join(config.MODEL_DIR, "trained", "consistent_stego_classifier_balancing.pkl")
    # trained_model_path = os.path.join(config.MODEL_DIR, "trained", "best_stego_classifier.pkl")
    
    if not os.path.exists(trained_model_path):
        print("Trained model not found. Please train the model first.")
        return
    
    from src.predict import StegoDetector, load_model_from_path
    detector = StegoDetector(trained_model_path)
    
    # Find all model files
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pth') or file.endswith('.pt'):
                model_files.append(os.path.join(root, file))
    
    print(f"Found {len(model_files)} model files to analyze...")
    
    results = []
    
    for model_file in tqdm(model_files, desc="Analyzing models"):
        try:
            # Determine model type
            if "resnet50" in model_file.lower():
                model_type = "resnet50"
            elif "mobilenet" in model_file.lower():
                model_type = "mobilenet_v3_small"
            else:
                model_type = "resnet50"  # Default
            
            # Load model
            model = load_model_from_path(model_file, model_type)
            
            # Predict
            result = detector.predict_single_model(model, os.path.basename(model_file))
            result['file_path'] = model_file
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {model_file}: {e}")
            results.append({
                'model_name': os.path.basename(model_file),
                'prediction': 'ERROR',
                'confidence': 0.0,
                'stego_probability': 0.0,
                'clean_probability': 0.0,
                'file_path': model_file,
                'error': str(e)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(config.DATA_DIR, "batch_analysis_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")
    
    # Print summary
    clean_count = len(results_df[results_df['prediction'] == 'CLEAN'])
    stego_count = len(results_df[results_df['prediction'] == 'STEGO'])
    error_count = len(results_df[results_df['prediction'] == 'ERROR'])
    
    print(f"\nBATCH ANALYSIS SUMMARY:")
    print(f"Total models analyzed: {len(results_df)}")
    print(f"Clean models: {clean_count}")
    print(f"Stego models: {stego_count}")
    print(f"Errors: {error_count}")
    
    # Show top suspicious models
    if stego_count > 0:
        print(f"\nTOP SUSPICIOUS MODELS (High stego probability):")
        suspicious = results_df[results_df['prediction'] == 'STEGO'].nlargest(5, 'stego_probability')
        for _, row in suspicious.iterrows():
            print(f"  {row['model_name']}: Stego prob = {row['stego_probability']:.4f}")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Batch analyze models in a directory')
#     parser.add_argument('directory', help='Directory containing model files')
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.directory):
#         print(f"Directory not found: {args.directory}")
#     else:
#         batch_analyze_models(args.directory)