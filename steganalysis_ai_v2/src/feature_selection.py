import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.config import config

class PSOFeatureSelection:
    def __init__(self, n_particles=config.PSO_N_PARTICLES, max_iter=config.PSO_MAX_ITER):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = config.PSO_W
        self.c1 = config.PSO_C1
        self.c2 = config.PSO_C2
        
    def initialize_particles(self, n_features):
        """Initialize particle positions and velocities"""
        positions = np.random.randint(0, 2, (self.n_particles, n_features))
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))
        return positions, velocities
    
    def fitness_function(self, X, y, particle):
        """Evaluate fitness using F1-score with feature subset"""
        selected_features = particle.astype(bool)
        
        if np.sum(selected_features) == 0:
            return 0.0  # Penalize no features selected
        
        X_subset = X[:, selected_features]
        
        # Use RandomForest with cross-validation
        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
            scores = cross_val_score(clf, X_subset, y, cv=3, scoring='f1')
            mean_f1 = np.mean(scores)
            
            # Add penalty for too many features
            feature_penalty = config.PSO_ALPHA * (np.sum(selected_features) / len(particle))
            fitness = mean_f1 - feature_penalty
            
            return max(fitness, 0)
        except:
            return 0.0
    
    def optimize(self, X, y):
        """Run PSO for feature selection"""
        n_samples, n_features = X.shape
        print(f"Starting PSO feature selection with {n_features} features...")
        
        # Initialize particles
        positions, velocities = self.initialize_particles(n_features)
        personal_best_positions = positions.copy()
        personal_best_scores = np.zeros(self.n_particles)
        global_best_position = None
        global_best_score = -np.inf
        
        # Evaluate initial particles
        for i in range(self.n_particles):
            score = self.fitness_function(X, y, positions[i])
            personal_best_scores[i] = score
            
            if score > global_best_score:
                global_best_score = score
                global_best_position = positions[i].copy()
        
        # PSO iterations
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               self.c2 * r2 * (global_best_position - positions[i]))
                
                # Update position with sigmoid transformation
                sigmoid_vel = 1 / (1 + np.exp(-velocities[i]))
                positions[i] = np.random.random(n_features) < sigmoid_vel
                positions[i] = positions[i].astype(int)
                
                # Evaluate new position
                current_score = self.fitness_function(X, y, positions[i])
                
                # Update personal best
                if current_score > personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i].copy()
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best F1: {global_best_score:.4f}")
        
        print(f"PSO completed. Best F1-score: {global_best_score:.4f}")
        print(f"Selected {np.sum(global_best_position)} out of {n_features} features")
        
        return global_best_position.astype(bool), global_best_score