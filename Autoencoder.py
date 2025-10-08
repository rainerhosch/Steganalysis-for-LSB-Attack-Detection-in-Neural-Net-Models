import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
class Autoencoder(nn.Module):
    def __init__(self, input_dim=256, encoding_dim=16):
        super().__init__()
        # Encoder yang sangat ringkas
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class WeightDataset(Dataset):
    def __init__(self, weight_data, seq_length=256, max_samples=10000):
        """
        Dataset untuk bobot model CNN
        Args:
            weight_data: List atau array dari bobot model
            seq_length: Panjang sequence untuk setiap sample
            max_samples: Maximum number of samples to create
        """
        self.data = []
        self.seq_length = seq_length
        self.max_samples = max_samples
        
        print(f"Processing weight data with max_samples: {max_samples}")
        
        # Process weights dengan memory efficiency
        total_sequences = 0
        for i, weights in enumerate(weight_data):
            if len(weights) >= seq_length:
                # Hitung berapa sequences yang bisa dibuat dari weights ini
                available_sequences = len(weights) // seq_length
                # Batasi jumlah sequences per weight array
                sequences_to_take = min(available_sequences, max_samples // (len(weight_data) or 1))
                
                print(f"Weight array {i}: {len(weights)} weights -> {sequences_to_take} sequences")
                
                for j in range(sequences_to_take):
                    start_idx = j * seq_length
                    end_idx = start_idx + seq_length
                    sequence = weights[start_idx:end_idx].astype(np.float32)
                    self.data.append(sequence)
                    total_sequences += 1
                    
                    # Batasi total samples
                    if len(self.data) >= max_samples:
                        print(f"Reached max_samples limit: {max_samples}")
                        break
            
            # Break jika sudah mencapai max_samples
            if len(self.data) >= max_samples:
                break
        
        self.data = np.array(self.data, dtype=np.float32)
        print(f"Dataset created: {len(self.data)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        # Normalisasi yang lebih stabil
        x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-8)
        return torch.FloatTensor(x_normalized), torch.FloatTensor(x_normalized)