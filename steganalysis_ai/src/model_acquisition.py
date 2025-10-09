import torch
import torchvision.models as models
import tensorflow as tf
import tensorflow_hub as hub
import os
from config import Config

class ModelAcquisition:
    def __init__(self):
        self.config = Config()
        
    def download_pytorch_models(self):
        """Mengunduh model PyTorch pre-trained"""
        models_dict = {}
        
        if 'resnet50' in self.config.MODEL_NAMES:
            models_dict['resnet50'] = models.resnet50(pretrained=True)
            print("Downloaded ResNet50")
            
        if 'mobilenet_v3_small' in self.config.MODEL_NAMES:
            models_dict['mobilenet_v3_small'] = models.mobilenet_v3_small(pretrained=True)
            print("Downloaded MobileNetV3 Small")
            
        return models_dict
    
    def save_model_weights(self, model, model_name, format='pytorch'):
        """Menyimpan bobot model"""
        os.makedirs(f"{self.config.DATA_PATH}models/", exist_ok=True)
        
        if format == 'pytorch':
            torch.save(model.state_dict(), f"{self.config.DATA_PATH}models/{model_name}_weights.pth", )
        elif format == 'tensorflow':
            model.save_weights(f"{self.config.DATA_PATH}models/{model_name}_weights.h5")
            
        print(f"Saved weights for {model_name}")
    
    def load_model_weights(self, model_name, model_architecture, format='pytorch'):
        """Memuat bobot model"""
        if format == 'pytorch':
            weights_path = f"{self.config.DATA_PATH}models/{model_name}_weights.pth"
            model_architecture.load_state_dict(torch.load(weights_path))
            return model_architecture