import torch.nn as nn
from scipy.stats import shannon_entropy
from sklearn.metrics import mean_squared_error
# Asumsikan Autoencoder (AE) dan Model Deteksi (D_model) sudah diimpor

# --- 1. Ekstraksi Loss & Gradient (Memerlukan Model DL) ---
def extract_loss_gradient(model: nn.Module, ae_model: nn.Module, data_input: torch.Tensor) -> tuple:
    """
    Menghitung Reconstruction Loss (Skalar) dan Gradient (Vektor/Skalar)
    """
    model.eval()
    
    # 1. Fitur Reconstruction Loss
    weight_tensor_flat = torch.cat([w.flatten() for w in model.parameters()])
    reconstructed_weights = ae_model(weight_tensor_flat)
    loss_reconstruction = mean_squared_error(weight_tensor_flat.detach().cpu().numpy(), 
                                            reconstructed_weights.detach().cpu().numpy())
    
    # 2. Fitur Gradient
    # Asumsi: Menggunakan CrossEntropyLoss pada satu sampel input
    target = torch.tensor([0]) # Placeholder target class
    criterion = nn.CrossEntropyLoss()
    
    # Hitung loss dan backpropagate untuk mendapatkan gradien
    output = model(data_input)
    loss_grad = criterion(output, target)
    loss_grad.backward()
    
    # Ekstraksi Gradien (ambil norm L2 sebagai fitur skalar, atau flatten sebagai vektor)
    gradients = [w.grad.flatten() for w in model.parameters() if w.grad is not None]
    gradient_vector = torch.cat(gradients).detach().cpu().numpy()
    
    # Menggunakan L2 norm sebagai fitur skalar untuk simplifikasi
    gradient_norm = np.linalg.norm(gradient_vector) 

    return loss_reconstruction, gradient_norm # Atau kembalikan gradient_vector jika PSO bisa menangani dimensi besar

# --- 2. Ekstraksi Entropi Bit-plane (Memerlukan NumPy) ---
def extract_bitplane_entropy(model_weights: dict) -> np.ndarray:
    """
    Menghitung Entropi Shannon untuk 23 bit-plane mantissa (float32)
    """
    entropy_vector = np.zeros(23)
    
    # Kumpulkan semua bobot float32 ke dalam satu array besar
    all_float_weights = np.concatenate([t.cpu().numpy().flatten() 
                                        for t in model_weights.values() 
                                        if t.dtype == torch.float32])
    
    # Konversi ke representasi integer 32-bit
    weights_int_view = all_float_weights.view(np.int32)
    
    # Analisis Bit-Plane (Mantissa adalah bit ke-0 hingga ke-22)
    for j in range(23):
        # Ekstraksi bit-plane j: (int_value >> j) & 1
        bit_plane = (weights_int_view >> j) & 1
        
        # Hitung probabilitas p0 dan p1
        p1 = np.mean(bit_plane)
        p0 = 1.0 - p1
        
        # Hitung Entropi Shannon H = - (p0*log2(p0) + p1*log2(p1))
        # Gunakan np.clip untuk menghindari log(0)
        h = shannon_entropy([p0, p1], base=2) 
        entropy_vector[j] = h
        
    return entropy_vector

# --- 3. Penggabungan Fitur ---
# F_combined = [loss_reconstruction] + [gradient_norm] + F_Ent
# F_combined adalah Vektor Fitur Berdimensi Tinggi yang menjadi input untuk PSO