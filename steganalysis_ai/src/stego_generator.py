import torch
import numpy as np
import struct
import os
from config import Config

class StegoGenerator:
    def __init__(self):
        self.config = Config()
        
    def float_to_binary(self, f):
        """Konversi float32 ke representasi biner"""
        return ''.join(bin(c).replace('0b', '').rjust(8, '0') 
                    for c in struct.pack('!f', f))
    
    def binary_to_float(self, b):
        """Konversi biner ke float32"""
        return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]
    
    def modify_lsb(self, weight_value, payload_bit, bit_position):
        """Memodifikasi LSB pada posisi tertentu"""
        if bit_position >= self.config.BIT_PLANES:  # Hanya mantissa (23 bit)
            return weight_value
            
        # Konversi ke biner
        binary_rep = self.float_to_binary(weight_value)
        
        # Pisahkan sign, exponent, mantissa
        sign_bit = binary_rep[0]
        exponent_bits = binary_rep[1:9]
        mantissa_bits = binary_rep[9:]
        
        # Modifikasi bit pada mantissa
        mantissa_list = list(mantissa_bits)
        if bit_position < len(mantissa_list):
            mantissa_list[bit_position] = str(payload_bit)
        
        modified_mantissa = ''.join(mantissa_list)
        modified_binary = sign_bit + exponent_bits + modified_mantissa
        
        # Konversi kembali ke float
        return self.binary_to_float(modified_binary)
    
    def generate_payload(self, size_ratio, total_weights):
        """Generate payload acak dengan ukuran tertentu"""
        payload_size = int(total_weights * size_ratio)
        return np.random.randint(0, 2, payload_size)
    
    def inject_payload(self, model_weights, embedding_rate, bit_position):
        """Menyisipkan payload ke dalam bobot model"""
        stego_weights = {}
        total_params = 0
        
        for name, param in model_weights.items():
            if param.dim() > 1:  # Hanya bobot, bukan bias
                flat_weights = param.flatten()
                total_params += flat_weights.numel()
        
        # Generate payload
        payload = self.generate_payload(embedding_rate, total_params)
        payload_idx = 0
        
        # Inject payload
        for name, param in model_weights.items():
            if param.dim() > 1:
                modified_param = param.clone()
                flat_weights = modified_param.flatten()
                
                for i in range(len(flat_weights)):
                    if payload_idx < len(payload):
                        if np.random.random() < embedding_rate:
                            new_value = self.modify_lsb(
                                flat_weights[i].item(), 
                                payload[payload_idx], 
                                bit_position
                            )
                            flat_weights[i] = new_value
                            payload_idx += 1
                
                stego_weights[name] = modified_param
            else:
                stego_weights[name] = param.clone()
                
        return stego_weights
    
    def create_stego_models(self, clean_models):
        """Membuat multiple stego models dengan variasi parameter"""
        stego_models = {}
        
        for model_name, model_weights in clean_models.items():
            stego_models[model_name] = {}
            
            for embedding_rate in self.config.EMBEDDING_RATES:
                for bit_position in self.config.BIT_POSITIONS[:self.config.BIT_INJECTED]:
                    key = f"{model_name}_rate_{embedding_rate}_bit_{bit_position}"
                    stego_models[model_name][key] = self.inject_payload(
                        model_weights, embedding_rate, bit_position
                    )
                    print(f"Created stego model: {key}")
                    
        return stego_models