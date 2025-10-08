from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# --- 1. Fungsi Kebugaran (Fitness Function) ---
def calculate_fitness(feature_subset_indices: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.01) -> float:
    """
    Menghitung Fitness (F1-Score - Complexity Penalty) dari sub-set fitur.
    """
    if len(feature_subset_indices) == 0:
        return 0.0
    
    # 1. Data Selection
    X_subset = X_train[:, feature_subset_indices]
    
    # 2. Train and Evaluate Classifier (Gunakan Random Forest atau XGBoost)
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Gunakan Cross-Validation untuk stabilitas
    f1_scores = cross_val_score(classifier, X_subset, y_train, cv=5, scoring='f1')
    f1_mean = np.mean(f1_scores)
    
    # 3. Complexity Penalty
    num_selected_features = len(feature_subset_indices)
    total_features = X_train.shape[1]
    
    # Complexity Penalty = alpha * (1 - (Num_Selected / Total_Features))
    complexity_penalty = alpha * (1 - (num_selected_features / total_features))
    
    # 4. Total Fitness: Maximize F1-Score while minimizing feature count
    # Perhatikan: F1-Score harus dimaksimalkan, penalty harus diminimalkan. 
    # Karena kita memaksimalkan fungsi fitness, kita bisa memaksimalkan F1-Score 
    # dan menambahkan faktor yang meningkat jika fitur lebih sedikit.
    fitness = f1_mean + complexity_penalty
    return fitness

# --- 2. Algoritma PSO Biner (Konsep Inti) ---
def run_pso_selection(X_train: np.ndarray, y_train: np.ndarray, n_particles: int = 30, max_iter: int = 100) -> np.ndarray:
    """
    Implementasi konseptual PSO Biner untuk Pemilihan Fitur.
    """
    D = X_train.shape[1] # Dimensi fitur
    # ... (Inisialisasi Partikel, Kecepatan, Pbest, Gbest) ...

    Gbest_features = np.zeros(D) # Vektor fitur terbaik global
    Gbest_fitness = -1.0 

    for iteration in range(max_iter):
        for i in range(n_particles):
            # 1. Update Kecepatan (V) menggunakan rumus PSO konvensional
            # ...

            # 2. Update Posisi (X) menggunakan fungsi Sigmoid dan Konversi Biner
            # ... 

            # 3. Hitung Fitness partikel saat ini
            selected_indices = np.where(X_i == 1)[0]
            current_fitness = calculate_fitness(selected_indices, X_train, y_train)

            # 4. Update Pbest dan Gbest
            # ...

        # Tampilkan Gbest_fitness per iterasi (untuk analisis konvergensi)
        print(f"Iterasi {iteration}: Gbest Fitness = {Gbest_fitness:.4f}")

    # Hasil akhir adalah Gbest_features yang berisi '1' untuk fitur yang optimal
    optimal_indices = np.where(Gbest_features == 1)[0]
    return optimal_indices

# --- 4. Pelatihan Model Akhir ---
# optimal_features = run_pso_selection(X_train, y_train)
# X_train_final = X_train[:, optimal_features]
# final_classifier = RandomForestClassifier(n_estimators=100)
# final_classifier.fit(X_train_final, y_train)
# ... (Evaluasi Model Akhir di Bab IV)