import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from config import Config

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FeatureExtractor:
    def __init__(self):
        self.config = Config()
        
    def extract_numerical_features(self, model_weights, test_data=None):
        """Mengekstrak fitur numerik (reconstruction loss dan gradients)"""
        features = {}
        
        # Reconstruction Loss
        reconstruction_loss = self.calculate_reconstruction_loss(model_weights)
        features['reconstruction_loss'] = reconstruction_loss
        
        # Gradients (jika test data tersedia)
        if test_data is not None:
            gradients = self.calculate_gradients(model_weights, test_data)
            features.update(gradients)
            
        return features
    
    def calculate_reconstruction_loss(self, model_weights):
        """Menghitung reconstruction loss menggunakan autoencoder"""
        # Flatten semua bobot
        all_weights = []
        for name, param in model_weights.items():
            if param.dim() > 1:
                all_weights.extend(param.flatten().tolist())
        
        all_weights = np.array(all_weights)
        input_dim = len(all_weights)
        
        # Inisialisasi dan train autoencoder
        autoencoder = Autoencoder(input_dim, self.config.AUTOENCODER_HIDDEN_DIM)
        
        # Training sederhana (dalam praktik nyata, butuh training yang proper)
        weights_tensor = torch.FloatTensor(all_weights).unsqueeze(0)
        reconstructed = autoencoder(weights_tensor)
        
        loss = nn.MSELoss()(reconstructed, weights_tensor)
        return loss.item()
    
    def calculate_gradients(self, model_weights, test_data):
        """Menghitung gradients melalui backpropagation"""
        # Implementasi perhitungan gradients
        # Ini adalah simplifikasi - dalam praktik butuh implementasi lengkap
        gradients = {
            'grad_mean': np.random.random(),  # Placeholder
            'grad_std': np.random.random(),   # Placeholder
            'grad_max': np.random.random()    # Placeholder
        }
        return gradients
    
    def calculate_bit_entropy(self, model_weights):
        """Menghitung entropi untuk setiap bit-plane"""
        entropies = {}
        
        for name, param in model_weights.items():
            if param.dim() > 1:
                flat_weights = param.flatten()
                
                for bit_pos in range(self.config.BIT_PLANES):
                    bit_values = []
                    
                    for weight in flat_weights:
                        binary_rep = self.float_to_binary(weight.item())
                        mantissa_bits = binary_rep[9:]  # Ambil mantissa
                        if bit_pos < len(mantissa_bits):
                            bit_value = int(mantissa_bits[bit_pos])
                            bit_values.append(bit_value)
                    
                    if bit_values:
                        entropy = self.calculate_shannon_entropy(bit_values)
                        entropies[f'bit_{bit_pos}_entropy'] = entropy
                        
        return entropies
    
    def float_to_binary(self, f):
        """Konversi float ke biner (helper function)"""
        import struct
        return ''.join(bin(c).replace('0b', '').rjust(8, '0') 
                      for c in struct.pack('!f', f))
    
    def calculate_shannon_entropy(self, bit_sequence):
        """Menghitung entropi Shannon untuk sequence bit"""
        from collections import Counter
        import math
        
        counts = Counter(bit_sequence)
        probabilities = [count / len(bit_sequence) for count in counts.values()]
        
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
                
        return entropy
    
    def extract_all_features(self, model_weights, test_data=None):
        """Mengekstrak semua fitur (numerical + entropy)"""
        features = {}
        
        # Fitur numerik
        numerical_features = self.extract_numerical_features(model_weights, test_data)
        features.update(numerical_features)
        
        # Fitur entropi
        entropy_features = self.calculate_bit_entropy(model_weights)
        features.update(entropy_features)
        
        return features