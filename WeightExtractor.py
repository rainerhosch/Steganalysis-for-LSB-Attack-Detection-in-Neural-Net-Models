import torch
import torch.nn as nn
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
import psutil
import os
from collections import OrderedDict

class WeightExtractor:
    def __init__(self, model_path_or_state_dict):
        # Cek jika input adalah path file atau state_dict langsung
        if isinstance(model_path_or_state_dict, str):
            loaded = torch.load(model_path_or_state_dict, map_location='cpu', weights_only=True)
        else:
            loaded = model_path_or_state_dict
        
        # Handle berbagai format model
        if isinstance(loaded, OrderedDict):
            self.state_dict = loaded
            print("Loaded state_dict directly")
        elif hasattr(loaded, 'state_dict'):
            self.state_dict = loaded.state_dict()
            print("Loaded model instance, extracted state_dict")
        elif isinstance(loaded, dict) and 'state_dict' in loaded:
            self.state_dict = loaded['state_dict']
            print("Loaded checkpoint, extracted state_dict")
        else:
            self.state_dict = loaded
            print("Treating as state_dict")
        
        print(f"Total keys in state_dict: {len(self.state_dict)}")
        
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def extract_conv_weights_only(self):
        """Ekstrak hanya bobot convolutional layers yang relevan untuk LSB steganalisis"""
        conv_weights = []
        layer_names = []
        
        # Pattern untuk convolutional layers
        conv_patterns = ['.conv1.weight', '.conv2.weight', '.conv3.weight', 
                        'downsample.0.weight', 'conv1.weight']
        
        for name, param in self.state_dict.items():
            if any(pattern in name for pattern in conv_patterns) and 'weight' in name:
                print(f"Extracting: {name}, Shape: {param.shape}")
                
                # Flatten bobot
                weight_vec = param.cpu().numpy().flatten()
                conv_weights.append(weight_vec)
                layer_names.append(name)
                
                # Clear memory setiap 5 layer
                if len(conv_weights) % 5 == 0:
                    gc.collect()
                    print(f"Memory usage: {self.get_memory_usage():.2f} MB")
        
        return conv_weights, layer_names
    
    def extract_conv_weights_by_layer(self, target_layers=None):
        """Ekstrak bobot per layer group untuk analisis lebih detail"""
        if target_layers is None:
            target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        
        layer_groups = {}
        
        for layer_group in target_layers:
            print(f"\nExtracting weights from {layer_group}...")
            group_weights = []
            
            for name, param in self.state_dict.items():
                if layer_group in name and any(x in name for x in ['.conv', 'downsample.0.weight']):
                    weight_vec = param.cpu().numpy().flatten()
                    group_weights.append(weight_vec)
            
            if group_weights:
                layer_groups[layer_group] = np.concatenate(group_weights)
                print(f"{layer_group}: {len(layer_groups[layer_group])} weights")
            
            gc.collect()
            print(f"Memory after {layer_group}: {self.get_memory_usage():.2f} MB")
        
        return layer_groups
    
    def get_layer_statistics(self):
        """Dapatkan statistik semua layer"""
        stats = {}
        for name, param in self.state_dict.items():
            stats[name] = {
                'shape': param.shape,
                'numel': param.numel(),
                'mean': param.float().mean().item(),
                'std': param.float().std().item(),
                'min': param.float().min().item(),
                'max': param.float().max().item()
            }
        return stats