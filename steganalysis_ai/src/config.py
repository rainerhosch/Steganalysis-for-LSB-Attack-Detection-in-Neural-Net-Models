import torch
import tensorflow as tf

class Config:
    # Model parameters
    MODEL_NAMES = ['resnet50', 'mobilenet_v3_small']
    MODEL_SOURCES = {
        'resnet50': 'pytorch/vision:v0.10.0',
        'mobilenet_v3_small': 'pytorch/vision:v0.10.0'
    }
    
    # Steganography parameters
    EMBEDDING_RATES = [0.001, 0.005, 0.01, 0.05, 0.1]  # 0.1% to 10%
    BIT_POSITIONS = list(range(23))  # All mantissa bits
    
    # Feature extraction
    BIT_PLANES = 23
    BIT_INJECTED = 3  # Hanya 3 LSB pertama
    AUTOENCODER_HIDDEN_DIM = 512
    
    # PSO parameters
    PSO_N_PARTICLES = 30
    PSO_MAX_ITER = 100
    PSO_W = 0.7
    PSO_C1 = 2.0
    PSO_C2 = 2.0
    
    # Classification
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = "data/"
    RESULTS_PATH = "results/"