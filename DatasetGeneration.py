import numpy as np
import torch # Ganti dengan tensorflow jika menggunakan TF/Keras
import random

def inject_lsb(model_weights: dict, payload_size_bits: int, target_bit: int) -> dict:
    """
    Menyuntikkan payload biner acak ke posisi bit LSB yang ditentukan 
    (misalnya, target_bit=0 untuk LSB pertama mantissa).
    """
    stego_weights = {}
    payload = np.random.randint(0, 2, payload_size_bits) # Payload biner acak
    payload_index = 0

    for name, tensor in model_weights.items():
        if tensor.dtype != torch.float32: # Pastikan hanya bobot float32 yang dimodifikasi
            stego_weights[name] = tensor
            continue
        
        # Konversi tensor float32 ke representasi biner (NumPy diperlukan)
        weights_flat = tensor.cpu().numpy().flatten()
        weights_view = weights_flat.view(np.int32) 
        
        # Masking dan penyuntikan LSB
        for i in range(len(weights_view)):
            if payload_index < payload_size_bits:
                # Dapatkan nilai integer 32-bit dari float
                int_val = weights_view[i]
                
                # Buat mask untuk menargetkan bit ke-target_bit (misal: 1 << target_bit)
                mask = 1 << target_bit
                
                # Hapus bit lama dan sisipkan bit payload baru
                # Posisi LSB untuk mantissa float32 adalah 0 sampai 22
                
                new_bit = payload[payload_index]
                
                # Hapus bit lama:
                int_val &= ~mask
                
                # Sisipkan bit baru:
                int_val |= (new_bit << target_bit)
                
                weights_view[i] = int_val
                payload_index += 1
            else:
                break

        # Konversi kembali ke tensor dan simpan
        weights_flat_new = weights_view.view(np.float32)
        stego_weights[name] = torch.from_numpy(weights_flat_new.reshape(tensor.shape)).to(tensor.device)
        
        if payload_index >= payload_size_bits:
            break

    return stego_weights, payload_index

# Contoh Penggunaan:
# weights_cover = load_model_weights("ResNet50_clean.pth")
# weights_stego, bits_injected = inject_lsb(weights_cover, payload_size_bits=100000, target_bit=0)
# save_model_weights(weights_stego, "ResNet50_stego_0_100k.pth")