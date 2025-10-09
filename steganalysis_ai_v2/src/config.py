# import torch
# import os

# # Konfigurasi dasar
# class Config:
#     # Path
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     MODEL_DIR = os.path.join(BASE_DIR, "models")
#     DATA_DIR = os.path.join(BASE_DIR, "data")
    
#     # Model settings
#     TARGET_MODELS = [
#         "resnet50",
#         "mobilenet_v3_small"
#     ]
    
#     # Steganography settings
#     PAYLOAD_SIZES = [0.001]  # EMBEDDING RATES 0.1% to 10% of capacity
#     BIT_POSITIONS = list(range(23))  # LSB positions of All mantissa bits
#     # BIT_POSITIONS = [0, 1, 2, 3]  # LSB positions to target
    
#     # Feature extraction
#     BIT_PLANES = 23
#     # BIT_INJECTED = 3  # Hanya 3 LSB pertama
#     # NUM_BIT_PLANES = 8  # Reduced for memory efficiency
#     BATCH_SIZE = 32     # Reduced batch size for 16GB RAM
    
#     # PSO settings
#     PSO_N_PARTICLES = 20  # Reduced for efficiency
#     PSO_MAX_ITER = 50
#     PSO_W = 0.7
#     PSO_C1 = 2.0
#     PSO_C2 = 2.0
    
#     # Training
#     TEST_SIZE = 0.2
#     RANDOM_STATE = 42
    
#     # Device
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {DEVICE}")

# config = Config()


import torch
import os

class Config:
    def __init__(self):
        # Path
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        
        # Model settings
        self.TARGET_MODELS = ["resnet50", "mobilenet_v3_small"]
        
        # Steganography settings
        self.PAYLOAD_SIZES = [0.001, 0.005, 0.01, 0.05, 0.1] # EMBEDDING RATES 0.1% to 10% of capacity
        self.BIT_POSITIONS = list(range(23)) # LSB positions of All mantissa bits
        # self.BIT_POSITIONS = [0, 1, 2, 3]
        self.BIT_PLANES = 23  # For entropy calculation
        
        # Feature extraction
        self.NUM_BIT_PLANES = 4  # Reduced for efficiency
        self.BATCH_SIZE = 8      # Smaller batch size
        
        # PSO settings
        self.PSO_N_PARTICLES = 10
        self.PSO_MAX_ITER = 20
        self.PSO_W = 0.7
        self.PSO_C1 = 2.0
        self.PSO_C2 = 2.0
        self.PSO_ALPHA = 0.01
        
        # Training
        self.TEST_SIZE = 0.3
        self.RANDOM_STATE = 42
        
        # Device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.DEVICE}")

# Global config instance
config = Config()