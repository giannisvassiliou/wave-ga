"""
Ultra-Fast WaveGA with Numba JIT Compilation
Target: 20-50x speedup over original implementation

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
def numba_balanced_accuracy_binary(y_true, y_pred):
    """Ultra-fast binary balanced accuracy."""
    n = len(y_true)
    if n == 0:
        return 0.5
    
    tp = fp = tn = fn = 0
    for i in range(n):
        if y_true[i] == 1:
            tp += (y_pred[i] == 1)
            fn += (y_pred[i] == 0)
        else:
            fp += (y_pred[i] == 1)
            tn += (y_pred[i] == 0)
    
    pos = tp + fn
    neg = tn + fp
    
    sensitivity = tp / pos if pos > 0 else 0.0
    specificity = tn / neg if neg > 0 else 0.0
    
    return (sensitivity + specificity) / 2.0


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
def numba_evaluate_population(weights_3d, X, y, target_class, n_classes, alpha):
    """
    Fully parallelized population fitness evaluation.
    
    Args:
        weights_3d: (pop_size, n_classes, n_features)
        X: (n_samples, n_features)
        y: (n_samples,)
        target_class: int
        n_classes: int
        alpha: float (blending weight)
    
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
        
        # Class-focused BA (binary)
        y_binary = np.zeros(n_samples, dtype=np.int32)
        y_pred_binary = np.zeros(n_samples, dtype=np.int32)
        
        for s in range(n_samples):
            y_binary[s] = (y[s] == target_class)
            y_pred_binary[s] = (y_pred[s] == target_class)
        
        f_class = numba_balanced_accuracy_binary(y_binary, y_pred_binary)
        
        # Overall multiclass BA
        f_overall = numba_balanced_accuracy_multiclass(y, y_pred, n_classes)
        
        # Blended fitness
        fitnesses[i] = alpha * f_class + (1.0 - alpha) * f_overall
    
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
# MAIN ULTRA-FAST WAVEGA CLASS
# =====================================================================

class UltraFastWaveGA:
    """
    Ultra-optimized WaveGA with Numba JIT compilation.
    Expected: 20-50x speedup over original implementation.
    """
    
    def __init__(self,
                 n_features,
                 population_size=30,
                 generations_per_wave=10,
                 n_cycles=3,
                 mutation_rate=0.1,
                 mutation_strength=0.1,
                 buffer_size=1000,
                 alpha=0.7,
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
        self.alpha = alpha
        self.crossover_type = crossover_type
        self.buffer_size = buffer_size
        
        # Population as 3D array: (pop_size, n_classes, n_features)
        self.population_weights = None
        self.population_fitness = None
        
        self.best_weights = None
        self.seen_classes = []
        self.n_classes = 0
        
        # Circular buffers (dict of arrays)
        self.class_buffers_X = {}  # class -> (buffer_size, n_features)
        self.class_buffers_size = {}  # class -> int (current size)
        self.class_buffers_ptr = {}  # class -> int (write pointer)
        
        # Stats
        self.training_time = 0.0
        self.n_updates = 0
        
        # Pre-compute random arrays for efficiency
        self._rng_cache_size = 10000
        self._rng_uniform_cache = np.random.rand(self._rng_cache_size)
        self._rng_normal_cache = np.random.randn(self._rng_cache_size)
        self._rng_cache_idx = 0
    
    def _get_random_uniform(self, n=1):
        """Fast random uniform from cache."""
        if self._rng_cache_idx + n > self._rng_cache_size:
            self._rng_uniform_cache = np.random.rand(self._rng_cache_size)
            self._rng_cache_idx = 0
        
        result = self._rng_uniform_cache[self._rng_cache_idx:self._rng_cache_idx + n]
        self._rng_cache_idx += n
        return result if n > 1 else result[0]
    
    def _initialize_population(self, X, y):
        """Initialize population with class-aware seeding."""
        self.population_weights = np.random.randn(
            self.population_size, self.n_classes, self.n_features
        ).astype(np.float32) * 0.1
        
        self.population_fitness = np.zeros(self.population_size, dtype=np.float32)
        
        # Seed first 30% with class centroids
        unique_classes = np.unique(y)
        n_seed = int(self.population_size * 0.3)
        
        for i in range(n_seed):
            for c in unique_classes:
                X_c = X[y == c]
                if len(X_c) > 0:
                    sample = X_c[np.random.randint(len(X_c))]
                    norm = np.linalg.norm(sample)
                    if norm > 1e-8:
                        self.population_weights[i, c] = sample / norm
    
    def _update_class_buffer(self, c, X_c):
        """Update circular buffer for class c."""
        n_samples = len(X_c)
        if n_samples == 0:
            return
        
        # Initialize buffer if needed
        if c not in self.class_buffers_X:
            self.class_buffers_X[c] = np.zeros(
                (self.buffer_size, self.n_features), dtype=np.float32
            )
            self.class_buffers_size[c] = 0
            self.class_buffers_ptr[c] = 0
        
        buffer = self.class_buffers_X[c]
        ptr = self.class_buffers_ptr[c]
        size = self.class_buffers_size[c]
        
        if n_samples >= self.buffer_size:
            # Replace entire buffer
            buffer[:] = X_c[-self.buffer_size:]
            self.class_buffers_size[c] = self.buffer_size
            self.class_buffers_ptr[c] = 0
        else:
            # Add circularly
            end_ptr = ptr + n_samples
            if end_ptr <= self.buffer_size:
                buffer[ptr:end_ptr] = X_c
            else:
                overflow = end_ptr - self.buffer_size
                buffer[ptr:] = X_c[:self.buffer_size - ptr]
                buffer[:overflow] = X_c[self.buffer_size - ptr:]
            
            self.class_buffers_ptr[c] = end_ptr % self.buffer_size
            self.class_buffers_size[c] = min(size + n_samples, self.buffer_size)
    
    def _sample_from_buffer(self, c, n):
        """Sample n examples from class c buffer."""
        if c not in self.class_buffers_X:
            return np.zeros((0, self.n_features), dtype=np.float32)
        
        size = self.class_buffers_size[c]
        if size == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)
        
        n = min(n, size)
        indices = np.random.choice(size, size=n, replace=False)
        
        buffer = self.class_buffers_X[c]
        ptr = self.class_buffers_ptr[c]
        
        if size < self.buffer_size:
            return buffer[indices].copy()
        else:
            # Adjust for circular buffer
            actual_indices = (ptr + indices) % self.buffer_size
            return buffer[actual_indices].copy()
    
    def _build_wave_batch(self, target_class, classes, n_samples=250):
        """Fast wave batch construction."""
        if target_class not in self.class_buffers_X:
            return None, None
        
        if self.class_buffers_size[target_class] < 10:
            return None, None
        
        # Sample target class (50%)
        n_target = n_samples // 2
        X_target = self._sample_from_buffer(target_class, n_target)
        y_target = np.full(len(X_target), target_class, dtype=np.int32)
        
        # Sample other classes (50%)
        other_classes = [c for c in classes if c != target_class 
                        and c in self.class_buffers_X 
                        and self.class_buffers_size[c] > 0]
        
        if not other_classes:
            return X_target, y_target
        
        n_others = n_samples - len(X_target)
        n_per_class = max(1, n_others // len(other_classes))
        
        X_list = [X_target]
        y_list = [y_target]
        
        for c in other_classes:
            X_c = self._sample_from_buffer(c, n_per_class)
            if len(X_c) > 0:
                y_c = np.full(len(X_c), c, dtype=np.int32)
                X_list.append(X_c)
                y_list.append(y_c)
        
        X_wave = np.vstack(X_list)
        y_wave = np.concatenate(y_list)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_wave))
        return X_wave[shuffle_idx], y_wave[shuffle_idx]
    
    def _wave_training(self, X, y, target_class, n_generations):
        """Numba-accelerated wave training."""
        for gen in range(n_generations):
            # Evaluate population using Numba
            self.population_fitness = numba_evaluate_population(
                self.population_weights, X, y, target_class, 
                self.n_classes, self.alpha
            )
            
            # Sort by fitness
            sorted_indices = np.argsort(self.population_fitness)[::-1]
            
            # # Elitism: keep top 2
            # new_population = np.zeros_like(self.population_weights)
            # new_population[0] = self.population_weights[sorted_indices[0]].copy()
            # new_population[1] = self.population_weights[sorted_indices[1]].copy()
            #

            # Elitism: keep top 15%
            n_elite = max(2, int(0.15 * self.population_size))  # At least 2, or 15% of population
            new_population = np.zeros_like(self.population_weights)
            
            for i in range(n_elite):
                new_population[i] = self.population_weights[sorted_indices[i]].copy()
            
            # Generate offspring
            offspring_idx = n_elite
            # Generate offspring
            # offspring_idx = 2
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
        """Ultra-fast incremental update."""
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
        
        # Update buffers
        for c in unique_classes:
            X_c = X[y == c]
            self._update_class_buffer(c, X_c)
        
        # Adaptive parameters
        total_buffer_size = sum(self.class_buffers_size.values())
        if total_buffer_size < 200:
            effective_cycles = 1
        elif total_buffer_size < 500:
            effective_cycles = 2
        else:
            effective_cycles = self.n_cycles
        
        n_active = sum(1 for c in self.seen_classes 
                      if c in self.class_buffers_X and self.class_buffers_size[c] > 10)
        
        effective_gens = max(3, self.generations_per_wave // 2) if n_active <= 1 else self.generations_per_wave
        
        # Wave training
        for _ in range(effective_cycles):
            for target_class in self.seen_classes:
                X_wave, y_wave = self._build_wave_batch(target_class, self.seen_classes)
                
                if X_wave is not None and len(X_wave) > 10:
                    self._wave_training(X_wave, y_wave, target_class, effective_gens)
        
        # Update best
        final_fitness = numba_evaluate_population(
            self.population_weights, X, y, 0, self.n_classes, 0.5
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


