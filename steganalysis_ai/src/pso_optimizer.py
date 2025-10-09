import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from config import Config

class PSOFeatureSelector:
    def __init__(self, features, labels):
        self.config = Config()
        self.features = features
        self.labels = labels
        self.n_features = features.shape[1]
        
        # Split data untuk evaluasi fitness
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
    
    def initialize_particles(self):
        """Inisialisasi populasi partikel"""
        particles = np.random.randint(0, 2, 
                                    (self.config.PSO_N_PARTICLES, self.n_features))
        velocities = np.zeros((self.config.PSO_N_PARTICLES, self.n_features))
        return particles, velocities
    
    def fitness_function(self, particle):
        """Fungsi fitness berdasarkan F1-Score"""
        selected_features = particle.astype(bool)
        
        if np.sum(selected_features) == 0:
            return 0.0
            
        # Train classifier dengan fitur terpilih
        classifier = RandomForestClassifier(n_estimators=50, random_state=self.config.RANDOM_STATE)
        classifier.fit(self.X_train[:, selected_features], self.y_train)
        
        # Predict dan hitung F1-Score
        y_pred = classifier.predict(self.X_test[:, selected_features])
        f1 = f1_score(self.y_test, y_pred)
        
        # Penalty untuk jumlah fitur yang banyak
        feature_penalty = self.config.PSO_ALPHA * (self.n_features - np.sum(selected_features)) / self.n_features
        
        return f1 + feature_penalty
    
    def optimize(self):
        """Algoritma PSO utama"""
        particles, velocities = self.initialize_particles()
        personal_best = particles.copy()
        personal_best_scores = np.array([self.fitness_function(p) for p in particles])
        
        global_best_idx = np.argmax(personal_best_scores)
        global_best = particles[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        for iteration in range(self.config.PSO_MAX_ITER):
            for i in range(self.config.PSO_N_PARTICLES):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (self.config.PSO_W * velocities[i] + 
                               self.config.PSO_C1 * r1 * (personal_best[i] - particles[i]) +
                               self.config.PSO_C2 * r2 * (global_best - particles[i]))
                
                # Update position (binary PSO)
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = np.random.random(self.n_features) < sigmoid
                particles[i] = particles[i].astype(int)
                
                # Evaluate fitness
                current_fitness = self.fitness_function(particles[i])
                
                # Update personal best
                if current_fitness > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = current_fitness
                    
                    # Update global best
                    if current_fitness > global_best_score:
                        global_best = particles[i].copy()
                        global_best_score = current_fitness
            
            print(f"Iteration {iteration + 1}, Best F1-Score: {global_best_score:.4f}")
            
        return global_best, global_best_score