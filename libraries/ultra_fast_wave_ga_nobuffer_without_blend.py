"""
Ultra-Fast WaveGA with Numba JIT Compilation - NO BUFFER, NO BLENDING VERSION
Uses OVERALL fitness only (multiclass balanced accuracy).
Trains directly on incoming data without buffer storage.

Prerequisites:
    pip install numba

This version uses Numba to compile critical hot paths to machine code.
"""

import time
import numpy as np
from numba import jit, prange
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# =====================================================================

@jit(nopython=True, cache=True, fastmath=True)
def numba_balanced_accuracy_multiclass(y_true, y_pred, n_classes):
    """Ultra-fast multiclass balanced accuracy."""
    n = len(y_true)
    if n == 0:
        return 0.5
    
    recalls = np.zeros(n_classes, dtype=np.float32)
    
    for c in range(n_classes):
        n_true = 0
        n_correct = 0
        
        for i in range(n):
            if y_true[i] == c:
                n_true += 1
                if y_pred[i] == c:
                    n_correct += 1
        
        if n_true > 0:
            recalls[c] = n_correct / n_true
        else:
            recalls[c] = 0.0
    
    return np.mean(recalls)


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def numba_evaluate_population(weights_3d, X, y, n_classes):
    """
    Fully parallelized population fitness evaluation with OVERALL fitness only.
    No blending - just multiclass balanced accuracy.
    
    Args:
        weights_3d: (pop_size, n_classes, n_features)
        X: (n_samples, n_features)
        y: (n_samples,)
        n_classes: int
    
    Returns:
        fitnesses: (pop_size,)
    """
    pop_size = weights_3d.shape[0]
    n_samples = X.shape[0]
    
    fitnesses = np.zeros(pop_size, dtype=np.float32)
    
    # Parallel over population
    for i in prange(pop_size):
        # Compute predictions for this chromosome
        y_pred = np.zeros(n_samples, dtype=np.int32)
        
        for s in range(n_samples):
            max_score = -1e10
            max_class = 0
            
            for c in range(n_classes):
                score = 0.0
                for f in range(X.shape[1]):
                    score += weights_3d[i, c, f] * X[s, f]
                
                if score > max_score:
                    max_score = score
                    max_class = c
            
            y_pred[s] = max_class
        
        # Overall multiclass BA ONLY (no blending)
        fitnesses[i] = numba_balanced_accuracy_multiclass(y, y_pred, n_classes)
    
    return fitnesses


@jit(nopython=True, cache=True, fastmath=True)
def numba_arithmetic_crossover(weights1, weights2, alpha):
    """Vectorized arithmetic crossover."""
    return alpha * weights1 + (1.0 - alpha) * weights2


@jit(nopython=True, cache=True, fastmath=True)
def numba_uniform_crossover(weights1, weights2, mask):
    """Vectorized uniform crossover."""
    n_classes, n_features = weights1.shape
    child = np.empty_like(weights1)
    
    for i in range(n_classes):
        for j in range(n_features):
            if mask[i, j]:
                child[i, j] = weights1[i, j]
            else:
                child[i, j] = weights2[i, j]
    
    return child


@jit(nopython=True, cache=True, fastmath=True)
def numba_mutate(weights, mutation_mask, mutation_values):
    """In-place mutation."""
    n_classes, n_features = weights.shape
    
    for i in range(n_classes):
        for j in range(n_features):
            if mutation_mask[i, j]:
                weights[i, j] += mutation_values[i, j]


@jit(nopython=True, cache=True)
def numba_predict(X, weights):
    """Fast batch prediction."""
    n_samples = X.shape[0]
    n_classes = weights.shape[0]
    
    y_pred = np.zeros(n_samples, dtype=np.int32)
    
    for s in range(n_samples):
        max_score = -1e10
        max_class = 0
        
        for c in range(n_classes):
            score = 0.0
            for f in range(X.shape[1]):
                score += weights[c, f] * X[s, f]
            
            if score > max_score:
                max_score = score
                max_class = c
        
        y_pred[s] = max_class
    
    return y_pred


# =====================================================================
# MAIN ULTRA-FAST WAVEGA NO-BUFFER CLASS
# =====================================================================

class UltraFastWaveGANoBuffer:
    """
    Ultra-optimized WaveGA with Numba JIT compilation - NO BUFFER, NO BLENDING VERSION.
    Uses OVERALL fitness only (multiclass balanced accuracy).
    Trains directly on incoming data without buffer storage.
    Expected: 20-50x speedup over original implementation.
    """
    
    def __init__(self,
                 n_features,
                 population_size=30,
                 generations_per_wave=10,
                 n_cycles=3,
                 mutation_rate=0.1,
                 mutation_strength=0.1,
                 crossover_type="arithmetic",
                 seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_type = crossover_type
        
        # Internal state
        self.n_classes = 0
        self.seen_classes = []
        self.population_weights = None
        self.population_fitness = None
        self.best_weights = None
        
        # Statistics
        self.training_time = 0.0
        self.n_updates = 0
        
        # Random state for reproducibility
        self._random_state = np.random.RandomState(seed)
    
    def _get_random_uniform(self):
        """Thread-safe random number."""
        return self._random_state.rand()
    
    def _initialize_population(self, X, y):
        """Initialize population with small random weights."""
        self.population_weights = np.random.randn(
            self.population_size, self.n_classes, self.n_features
        ).astype(np.float32) * 0.01
        
        self.population_fitness = np.zeros(self.population_size, dtype=np.float32)
    
    def _build_wave_batch_from_data(self, X, y, target_class):
        """
        Build wave batch directly from incoming data (no buffer sampling).
        
        Strategy:
        - Extract all samples of target class
        - Extract all samples of other classes
        - Balance if needed
        """
        # Get target class samples
        target_mask = (y == target_class)
        X_target = X[target_mask]
        y_target = y[target_mask]
        
        # Get other class samples
        other_mask = ~target_mask
        X_other = X[other_mask]
        y_other = y[other_mask]
        
        # If no target class samples, skip this wave
        if len(X_target) == 0:
            return None, None
        
        # If no other class samples, return only target
        if len(X_other) == 0:
            return X_target, y_target
        
        # Combine target and other classes
        X_wave = np.vstack([X_target, X_other])
        y_wave = np.concatenate([y_target, y_other])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_wave))
        return X_wave[shuffle_idx], y_wave[shuffle_idx]
    
    def _wave_training(self, X, y, n_generations):
        """Numba-accelerated wave training with overall fitness only."""
        for gen in range(n_generations):
            # Evaluate population using Numba (OVERALL FITNESS ONLY)
            self.population_fitness = numba_evaluate_population(
                self.population_weights, X, y, self.n_classes
            )
            
            # Sort by fitness
            sorted_indices = np.argsort(self.population_fitness)[::-1]
            
            # Elitism: keep top 15%
            n_elite = max(2, int(0.15 * self.population_size))
            new_population = np.zeros_like(self.population_weights)
            
            for i in range(n_elite):
                new_population[i] = self.population_weights[sorted_indices[i]].copy()
            
            # Generate offspring
            offspring_idx = n_elite
            while offspring_idx < self.population_size:
                # Tournament selection
                t1_idx = np.random.choice(self.population_size, size=3, replace=False)
                t2_idx = np.random.choice(self.population_size, size=3, replace=False)
                
                parent1_idx = t1_idx[np.argmax(self.population_fitness[t1_idx])]
                parent2_idx = t2_idx[np.argmax(self.population_fitness[t2_idx])]
                
                parent1 = self.population_weights[parent1_idx]
                parent2 = self.population_weights[parent2_idx]
                
                # Crossover
                if self.crossover_type == "arithmetic":
                    alpha_cross = self._get_random_uniform()
                    alpha_cross = 0.3 + 0.4 * alpha_cross  # Map to [0.3, 0.7]
                    child = numba_arithmetic_crossover(parent1, parent2, alpha_cross)
                else:  # uniform
                    mask = np.random.rand(self.n_classes, self.n_features) < 0.5
                    child = numba_uniform_crossover(parent1, parent2, mask)
                
                # Mutation
                if self._get_random_uniform() < self.mutation_rate:
                    mutation_mask = np.random.rand(self.n_classes, self.n_features) < 0.1
                    mutation_values = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
                    numba_mutate(child, mutation_mask, mutation_values)
                
                new_population[offspring_idx] = child
                offspring_idx += 1
            
            self.population_weights = new_population
    
    def partial_fit(self, X, y):
        """Ultra-fast incremental update without buffer."""
        start_time = time.time()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        # Detect new classes
        unique_classes = np.unique(y)
        new_classes = set(unique_classes) - set(self.seen_classes)
        
        if new_classes:
            print(f"NEW CLASSES: {new_classes}")
            old_n_classes = self.n_classes
            
            for c in sorted(new_classes):
                self.seen_classes.append(c)
            self.seen_classes.sort()
            self.n_classes = len(self.seen_classes)
            
            # Expand population
            if self.population_weights is not None:
                old_pop = self.population_weights
                new_pop = np.zeros(
                    (self.population_size, self.n_classes, self.n_features),
                    dtype=np.float32
                )
                new_pop[:, :old_n_classes, :] = old_pop
                self.population_weights = new_pop
        else:
            if self.n_classes == 0:
                self.seen_classes = sorted(unique_classes)
                self.n_classes = len(self.seen_classes)
        
        # Initialize if needed
        if self.population_weights is None:
            self._initialize_population(X, y)
        
        # Adaptive parameters based on batch size
        batch_size = len(X)
        
        if batch_size < 50:
            effective_cycles = 1
            effective_gens = max(3, self.generations_per_wave // 2)
        elif batch_size < 150:
            effective_cycles = 2
            effective_gens = self.generations_per_wave
        else:
            effective_cycles = self.n_cycles
            effective_gens = self.generations_per_wave
        
        # Check how many classes are present in current batch
        classes_in_batch = unique_classes
        n_classes_in_batch = len(classes_in_batch)
        
        # Adjust generations if only one class present
        if n_classes_in_batch <= 1:
            effective_gens = max(3, self.generations_per_wave // 2)
        
        # Wave training: iterate through each class that appears in the data
        for _ in range(effective_cycles):
            for target_class in classes_in_batch:
                # Build wave batch from incoming data (not from buffer)
                X_wave, y_wave = self._build_wave_batch_from_data(X, y, target_class)
                
                if X_wave is not None and len(X_wave) > 5:
                    self._wave_training(X_wave, y_wave, effective_gens)
        
        # Update best weights using overall fitness on full batch
        final_fitness = numba_evaluate_population(
            self.population_weights, X, y, self.n_classes
        )
        best_idx = np.argmax(final_fitness)
        self.best_weights = self.population_weights[best_idx].copy()
        
        self.training_time += time.time() - start_time
        self.n_updates += 1
    
    def predict(self, X):
        """Fast prediction."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_weights is None:
            return np.zeros(len(X), dtype=np.int32)
        return numba_predict(X, self.best_weights)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_weights is None:
            return np.ones((len(X), max(1, self.n_classes))) / max(1, self.n_classes)
        
        scores = X @ self.best_weights.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
