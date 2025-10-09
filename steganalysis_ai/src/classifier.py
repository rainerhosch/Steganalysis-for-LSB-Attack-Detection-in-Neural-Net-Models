import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from config import Config

class StegoClassifier:
    def __init__(self):
        self.config = Config()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE),
            'xgboost': XGBClassifier(random_state=self.config.RANDOM_STATE),
            'svm': SVC(random_state=self.config.RANDOM_STATE)
        }
    
    def prepare_dataset(self, cover_features, stego_features):
        """Mempersiapkan dataset untuk training"""
        X = cover_features + stego_features
        y = [0] * len(cover_features) + [1] * len(stego_features)  # 0: cover, 1: stego
        
        return np.array(X), np.array(y)
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluasi model dengan berbagai metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics, confusion_matrix(y_test, y_pred)
    
    def train_and_evaluate(self, X, y, selected_features=None):
        """Training dan evaluasi semua model"""
        if selected_features is not None:
            X = X[:, selected_features]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            metrics, cm = self.evaluate_model(model, X_test, y_test)
            results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'model': model
            }
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def save_models(self, results, path):
        """Menyimpan model yang telah dilatih"""
        for model_name, result in results.items():
            joblib.dump(result['model'], f"{path}{model_name}.joblib")
    
    def load_model(self, model_name, path):
        """Memuat model yang telah disimpan"""
        return joblib.load(f"{path}{model_name}.joblib")