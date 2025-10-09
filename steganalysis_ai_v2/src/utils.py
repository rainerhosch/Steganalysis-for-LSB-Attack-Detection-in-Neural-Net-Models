import numpy as np
import torch
import struct
from typing import List, Tuple

def float_to_bin32(f: float) -> str:
    """Convert float32 to binary representation"""
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

def bin32_to_float(b: str) -> float:
    """Convert binary representation to float32"""
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]

def extract_mantissa_bits(f: float, num_bits: int = 8) -> List[int]:
    """Extract specified number of mantissa bits from float32"""
    binary = float_to_bin32(f)
    # Mantissa bits are from position 9 to 31 (23 bits)
    mantissa_bits = binary[9:9+num_bits]
    return [int(bit) for bit in mantissa_bits]

def calculate_shannon_entropy(bit_sequence: List[int]) -> float:
    """Calculate Shannon entropy for a bit sequence"""
    if len(bit_sequence) == 0:
        return 0.0
    
    p1 = sum(bit_sequence) / len(bit_sequence)
    p0 = 1 - p1
    
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    
    return entropy

def memory_usage_monitor():
    """Monitor memory usage to prevent OOM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")