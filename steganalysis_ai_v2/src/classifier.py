import datetime as datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from src.config import config

class StegoClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
            'svm': SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
        }
        self.best_model = None
        self.feature_names = None
        
    def prepare_data(self, features_df):
        """Prepare data for training"""
        # Separate features and target
        feature_columns = [col for col in features_df.columns if col not in 
                        ['model_name', 'model_type', 'is_stego']]
        X = features_df[feature_columns].values
        y = features_df['is_stego'].values
        
        self.feature_names = feature_columns
        return X, y
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
        
        return metrics
    
    def train_and_evaluate(self, features_df, feature_mask=None):
        """Train and evaluate all models"""
        X, y = self.prepare_data(features_df)
        
        # Apply feature selection if mask provided
        if feature_mask is not None:
            X = X[:, feature_mask]
            selected_features = [self.feature_names[i] for i in range(len(feature_mask)) if feature_mask[i]]
            print(f"Using {len(selected_features)} selected features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results[name] = metrics
            
            # Save the best model based on F1-score
            if self.best_model is None or metrics['f1_score'] > results.get(self.best_model, {}).get('f1_score', 0):
                self.best_model = name
        
        print(f"\nBest model: {self.best_model}")
        return results, self.models[self.best_model]
    
    def save_model(self, model, filename="best_stego_classifier.pkl"):
        """Save trained model"""
        model_path = os.path.join(config.MODEL_DIR, "trained", filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename="best_stego_classifier.pkl"):
        """Load trained model"""
        model_path = os.path.join(config.MODEL_DIR, "trained", filename)
        return joblib.load(model_path)
    
    #Update
    # Di src/classifier.py - modifikasi method train_and_evaluate
    def train_and_evaluate_v2(self, features_df, feature_mask=None):
        """Train and evaluate all models - version with feature names return"""
        X, y = self.prepare_data(features_df)
        
        # Apply feature selection if mask provided
        if feature_mask is not None:
            X = X[:, feature_mask]
            selected_features = [self.feature_names[i] for i in range(len(feature_mask)) if feature_mask[i]]
            print(f"Using {len(selected_features)} selected features: {selected_features}")
            
            # Save selected feature names for later use
            self.selected_feature_names = selected_features
        else:
            self.selected_feature_names = self.feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results[name] = metrics
            
            # Save the best model based on F1-score
            if self.best_model is None or metrics['f1_score'] > results.get(self.best_model, {}).get('f1_score', 0):
                self.best_model = name
        
        print(f"\nBest model: {self.best_model}")
        
        # Return 3 values: results, best_model, feature_names
        return results, self.models[self.best_model], self.selected_feature_names

    def save_model_v2(self, model, feature_names, filename="best_stego_classifier.pkl"):
        """Save trained model and feature names"""
        model_path = os.path.join(config.MODEL_DIR, "trained", filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save both model and feature names
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'timestamp': datetime.datetime.now().isoformat()
        }
        joblib.dump(model_data, model_path)
        print(f"Model and feature names saved to {model_path}")