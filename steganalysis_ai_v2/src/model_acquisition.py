import torch
import torchvision.models as models
import os
from src.config import config
from src.utils import memory_usage_monitor

class ModelAcquisition:
    def __init__(self):
        self.model_dir = config.MODEL_DIR
        os.makedirs(os.path.join(self.model_dir, "cover"), exist_ok=True)
        
    def download_pretrained_models(self):
        """Download pre-trained models"""
        print("Downloading pre-trained models...")
        
        model_instances = {}
        
        # Use new weights API to avoid deprecation warnings
        if "resnet50" in config.TARGET_MODELS:
            print("Downloading ResNet50...")
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
            except ImportError:
                weights = "IMAGENET1K_V1"  # fallback for older torchvision
            model = models.resnet50(weights=weights)
            model_instances["resnet50"] = model
            torch.save(model.state_dict(), 
                    os.path.join(config.MODEL_DIR, "cover", "resnet50.pth"))
        
        if "mobilenet_v3_small" in config.TARGET_MODELS:
            print("Downloading MobileNetV3 Small...")
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.DEFAULT
            except ImportError:
                weights = "IMAGENET1K_V1"
            model = models.mobilenet_v3_small(weights=weights)
            model_instances["mobilenet_v3_small"] = model
            torch.save(model.state_dict(), 
                    os.path.join(config.MODEL_DIR, "cover", "mobilenet_v3_small.pth"))
        
        memory_usage_monitor()
        return model_instances
    
    def load_model(self, model_name: str):
        """Load specific model"""
        model_path = os.path.join(config.MODEL_DIR, "cover", f"{model_name}.pth")
        
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
        elif model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model